import numpy as np


def get_2d_sincos_positoinal_embedding(embedding_dimension, height, width):
  """Generates a 2D sin-cos positional embedding.

  Args:
    embedding_dimension: An integer, specifying the embedding dimension.
    height: An integer, specifying the height.
    width: An integer, specifying the width.

  Returns:
    A numpy array with positional embedding.
  """
  grid_h = np.arange(height, dtype=np.float32)
  grid_w = np.arange(width, dtype=np.float32)
  grid = np.meshgrid(grid_w, grid_h)
  grid = np.stack(grid, axis=0)

  grid = grid.reshape([2, height, width, 1])
  positoinal_embedding = get_2d_sincos_positoinal_embedding_from_grid(
      embedding_dimension, grid)
  return positoinal_embedding.reshape([1, height, width, embedding_dimension])

def get_2d_sincos_positoinal_embedding_from_grid(embedding_dimension, grid):
  """Generates a 2D sin-cos positional embedding given a coordinates grid.

  Args:
    embedding_dimension: An integer, specifying the embedding dimension.
    grid: A numpy array, with shape [2, height, width, 1], containing the
      2D coordinates.

  Returns:
    A numpy array with positional embedding.

  Raises:
    ValueError: If the embedding_dimension cannot be divided by 2.
  """
  if embedding_dimension % 2 != 0:
    raise ValueError('Expect the embedding_dimension to be divisable by 2!')

  positional_embedding_height = get_1d_sincos_positional_embedding_from_grid(
      embedding_dimension // 2, grid[0])
  positional_embedding_width = get_1d_sincos_positional_embedding_from_grid(
      embedding_dimension // 2, grid[1])

  positional_embedding = np.concatenate(
      [positional_embedding_height, positional_embedding_width], axis=-1)
  return positional_embedding


def get_1d_sincos_positional_embedding_from_grid(embedding_dimension, position):
  """Generates a 1D sin-cos positional embedding given a coordinates grid.

  Args:
    embedding_dimension: An integer, specifying the embedding dimension.
    position: A numpy array, with shape [height, width, 1], containing the
      position coordinates.

  Returns:
    A numpy array with positional embedding.

  Raises:
    ValueError: If the embedding_dimension cannot be divided by 2.
  """
  if embedding_dimension % 2 != 0:
    raise ValueError('Expect the embedding_dimension to be divisable by 2!')
  omega = np.arange(embedding_dimension // 2, dtype=float)
  omega /= embedding_dimension / 2.
  omega = 1. / 10000**omega

  position = position.reshape(-1)
  out = np.einsum('m,d->md', position, omega)

  position_embedding_sin = np.sin(out)
  position_embedding_cos = np.cos(out)

  position_embedding = np.concatenate(
      [position_embedding_sin, position_embedding_cos], axis=1)
  return position_embedding

class AddAbolustePositionalEncoding(nn.Module):

    def __init__(self, input_shape):
        positional_embedding_numpy = get_2d_sincos_positional_embedding()
