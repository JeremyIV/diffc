import numpy as np
from lib.diffc.rcc.pfr import reverse_channel_encode, reverse_channel_decode


def partition_mu(dim, chunk_sizes, shared_seed=0):
    """
    return an array of shape (dim,) which determines which chunk each dimension belongs to.
    the values in the array correspond to the indices of the chunk.
    """
    total_bits = sum(chunk_sizes)
    chunk_ndims = []
    for chunk_size in chunk_sizes[:-1]:
        chunk_ndims.append(int(dim * chunk_size / total_bits))
    chunk_ndims.append(dim - sum(chunk_ndims))

    # An empty chunk would still encode a seed (pure bitstream waste) while
    # its dimensions pile into the last chunk, far exceeding what that
    # chunk's 2^chunk_size candidates can represent.
    if min(chunk_ndims) < 1:
        raise ValueError(
            f"Cannot partition {dim} dimensions into {len(chunk_sizes)} "
            f"chunks: at least one chunk would receive zero dimensions. "
            f"The step's Dkl is too large for the latent dimension at "
            f"this max_chunk_size."
        )

    partition_indices = np.concatenate(
        [np.full(ndims, i) for i, ndims in enumerate(chunk_ndims)]
    )
    rng = np.random.default_rng(shared_seed)
    rng.shuffle(partition_indices)

    return partition_indices


def combine_partitions(partition_indices, partitions):
    combined = np.zeros_like(partition_indices, dtype=partitions[0].dtype)
    for i, partition in enumerate(partitions):
        combined[partition_indices == i] = partition
    return combined


def chunk_and_encode(mu, chunk_sizes, shared_seed=0):
    partition_indices = partition_mu(len(mu), chunk_sizes, shared_seed)

    partitions = []
    seeds = []
    for i, chunk_size in enumerate(chunk_sizes):
        chunk_mask = partition_indices == i
        mu_chunk = mu[chunk_mask]
        chunk_shared_seed = hash((shared_seed, i)) % (2 ** 32)
        seed, partition = reverse_channel_encode(
            mu_chunk, K=int(2 ** chunk_size), shared_seed=chunk_shared_seed
        )
        seeds.append(seed)
        partitions.append(partition)

    return tuple(seeds), combine_partitions(partition_indices, partitions)


def decode_from_chunks(dim, seeds, chunk_sizes, shared_seed=0):
    partition_indices = partition_mu(dim, chunk_sizes, shared_seed)

    partitions = []
    for i, seed in enumerate(seeds):
        chunk_shared_seed = hash((shared_seed, i)) % (2 ** 32)
        chunk_dim = (partition_indices == i).sum()
        partition = reverse_channel_decode(
            chunk_dim, seed, shared_seed=chunk_shared_seed
        )
        partitions.append(partition)
    return combine_partitions(partition_indices, partitions)


def distribute_apples(m, n):
    """
    Given m apples and n buckets, return how many apples to put in each bucket, to distribute as evenly as possible.
    """
    if n == 0:
        return []

    base_apples = m // n
    extra_apples = m % n

    distribution = [base_apples] * n

    for i in range(extra_apples):
        distribution[i] += 1

    return tuple(distribution)


def get_chunk_sizes(Dkl, max_size=8, chunk_padding_bits=2):
    n = int(np.ceil(Dkl))
    num_chunks = int(np.ceil(n / (max_size - chunk_padding_bits)))
    return distribute_apples(n + chunk_padding_bits * num_chunks, num_chunks)
