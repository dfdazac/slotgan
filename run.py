import os.path as osp

import torch.cuda
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, AdamW
from torch.optim import Adam
from tap import Tap
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
import mdmm

import data
from data import (get_dataset_pairs, TransformersCollator,
                  make_infinite_iterator)
from models import TokenEmbedding, EntityExtractor, ConvolutionalDiscriminator
from utils import make_mdmm_optimizer, get_gradient_penalty
import eval


class Arguments(Tap):
    dataset: data.DATASETS_LITERAL = data.DATASET_WIKIDATA5M

    base_model: str = 'bert-base-cased'
    num_slots: int = 10
    slot_iters: int = 3
    stretch: bool = True

    generator_iters: int = 20_000
    discriminator_iters: int = 5
    grad_penalty: float = 10.0
    drop_probability: float = 0.5
    shift_names: bool = True
    randomize_case: bool = False
    constraint_lr: float = 1e-3

    batch_size: int = 32
    evaluate: bool = True
    eval_batch_size: int = 512

    checkpoint: str = None
    num_workers: int = 2
    log_wandb: bool = True
    notes: str = None
    watch_gradients: bool = False


def train(args: Arguments):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wandb.init(project='adv',
               mode='online' if args.log_wandb else 'disabled',
               config=args.as_dict(), notes=args.notes)

    text_dataset, names_dataset = get_dataset_pairs(args.dataset)
    collator = TransformersCollator(args.base_model, max_length=128,
                                    shift_names=args.shift_names,
                                    randomize_case=args.randomize_case)

    text_loader = DataLoader(text_dataset,
                             batch_size=args.batch_size,
                             shuffle=True,
                             collate_fn=collator.collate_text_fn,
                             num_workers=args.num_workers,
                             pin_memory=True)
    text_sampler = make_infinite_iterator(text_loader)

    names_loader = DataLoader(names_dataset,
                              batch_size=args.batch_size * args.num_slots,
                              shuffle=True,
                              collate_fn=collator.collate_names_fn,
                              num_workers=args.num_workers,
                              pin_memory=True)
    names_sampler = make_infinite_iterator(names_loader)

    embedding = TokenEmbedding(args.base_model).to(device)

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
    else:
        checkpoint = None

    generator = EntityExtractor(args.base_model, args.num_slots,
                                num_iters=args.slot_iters,
                                stretch=args.stretch)
    if checkpoint is not None:
        generator.load_state_dict(checkpoint['generator'])
    generator = generator.to(device)

    expected_transitions = 0
    constraint = mdmm.MDMM([mdmm.EqConstraint(lambda: expected_transitions,
                                              value=2)])
    if checkpoint is not None:
        constraint.load_state_dict(checkpoint['constraint'])
    constraint = constraint.to(device)
    constraint_optimizer = make_mdmm_optimizer(constraint, args.constraint_lr,
                                               Adam)
    gen_optimizer = AdamW(generator.parameters(), lr=2e-5)
    gen_warmup = int(0.1 * args.generator_iters)
    gen_scheduler = get_linear_schedule_with_warmup(
        gen_optimizer,
        num_warmup_steps=gen_warmup,
        num_training_steps=args.generator_iters
    )

    discriminator = ConvolutionalDiscriminator(
        in_features=embedding.embedding_dim
    )

    if checkpoint is not None:
        discriminator.load_state_dict(checkpoint['discriminator'])
    discriminator = discriminator.to(device)
    disc_optimizer = AdamW(discriminator.parameters(), lr=2e-5)
    disc_iters = args.generator_iters * args.discriminator_iters
    disc_warmup = int(0.2 * disc_iters)
    disc_scheduler = get_linear_schedule_with_warmup(
        disc_optimizer,
        num_warmup_steps=disc_warmup,
        num_training_steps=disc_iters
    )

    evaluator = eval.CoNLLEvaluator('validation', collator.tokenizer,
                                    args.eval_batch_size, device)
    if args.watch_gradients:
        wandb.watch(generator, log_freq=100)

    figure = plt.figure()
    checkpoint_path = osp.join('models', f'{wandb.run.id}.pt')

    bar = tqdm(range(args.generator_iters), miniters=10, mininterval=1,
               desc='Training')
    for iteration in bar:
        for i in range(args.discriminator_iters):
            text_tokens, text_mask = next(text_sampler)
            batch_size = text_tokens.shape[0]

            with torch.no_grad():
                text_embeds = embedding(text_tokens.to(device))
                fake_samples, *_ = generator(text_embeds.to(device),
                                             text_mask.to(device))
                fake_samples = fake_samples.reshape(batch_size * args.num_slots,
                                                    -1,
                                                    embedding.embedding_dim)

            name_tokens, name_mask = next(names_sampler)
            name_tokens = name_tokens.to(device)
            name_mask = name_mask.to(device)

            real_samples = embedding(name_tokens) * name_mask.unsqueeze(-1)
            num_real_samples = real_samples.shape[0]
            drop_mask = torch.rand(num_real_samples) < args.drop_probability
            real_samples[drop_mask] = 0.0
            disc_real = discriminator(real_samples).mean()
            disc_fake = discriminator(fake_samples).mean()

            grad_penalty = get_gradient_penalty(discriminator,
                                                real_samples,
                                                fake_samples)

            disc_loss = disc_fake - disc_real + args.grad_penalty * grad_penalty

            disc_optimizer.zero_grad()
            disc_loss.backward()
            disc_optimizer.step()
            disc_scheduler.step()

        text_tokens, text_mask = next(text_sampler)
        batch_size = text_tokens.shape[0]

        text_embeds = embedding(text_tokens.to(device))
        fake_samples, slot_masks, full_masks = generator(text_embeds.to(device),
                                                         text_mask.to(device))
        fake_samples = fake_samples.reshape(batch_size * args.num_slots,
                                            -1,
                                            embedding.embedding_dim)

        flat_masks = slot_masks.reshape(batch_size * args.num_slots, -1)
        slot_indicators, _ = flat_masks.max(dim=-1, keepdim=True)
        active_slots = slot_indicators.sum()
        transitions = (flat_masks[:, 1:] - flat_masks[:, :-1]).abs()
        transitions = transitions.sum(dim=-1, keepdim=True) * slot_indicators
        transitions = transitions.sum()
        expected_transitions = transitions / (active_slots + 1e-9)

        gen_loss = -discriminator(fake_samples).mean()
        constrained_loss = constraint(gen_loss)

        gen_optimizer.zero_grad()
        constraint_optimizer.zero_grad()
        constrained_loss.value.backward()
        gen_optimizer.step()
        gen_scheduler.step()
        constraint_optimizer.step()

        wandb.log({'disc_real': disc_real.item(),
                   'disc_fake': disc_fake.item(),
                   'grad_penalty': grad_penalty.item(),
                   'disc_loss': disc_loss.item(),
                   'gen_loss': gen_loss.item(),
                   'constrained_loss': constrained_loss.value.item(),
                   'expected_transitions': expected_transitions.item(),
                   'lambda': constraint[0].lmbda.item(),
                   'infeasibility': constrained_loss.infs[0].item()})

        bar.update(1)

        if iteration % 100 == 0:
            if args.evaluate:
                wandb.log(evaluator.evaluate_marginal(embedding, generator))

            plt.clf()
            slot_sample = full_masks[0].detach().cpu().numpy().T
            length = text_mask[0].sum().item()
            slot_sample = slot_sample[:length]
            text_sample = text_tokens[0, :length]

            plt.imshow(slot_sample, vmin=0, vmax=1.0)
            plt.colorbar()
            tokens = collator.tokenizer.convert_ids_to_tokens(text_sample)
            plt.yticks(range(slot_sample.shape[0]), tokens)
            wandb.log({f'slot_mask_{iteration}': wandb.Image(figure)})

            torch.save({'discriminator': discriminator.state_dict(),
                        'generator': generator.state_dict(),
                        'constraint': constraint.state_dict()},
                       checkpoint_path)

    torch.save({'discriminator': discriminator.state_dict(),
                'generator': generator.state_dict(),
                'constraint': constraint.state_dict()},
               checkpoint_path)


if __name__ == '__main__':
    train(Arguments(explicit_bool=True).parse_args())
