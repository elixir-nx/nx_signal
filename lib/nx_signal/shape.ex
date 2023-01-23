defmodule NxSignal.Shape do
  # Conveniences for manipulating shapes internal to NxSignal.
  # Copied from Nx.Shape because we needed Nx.Shape.pool
  @moduledoc false

  @doc """
  Returns a padding configuration based on the given pad mode
  for the given input shape, kernel size and stride.

  By default, interior padding is not considered in the padding
  configuration.

  ## Examples

      iex> NxSignal.Shape.to_padding_config({2, 3, 2}, {2, 3, 2}, :valid)
      [{0, 0}, {0, 0}, {0, 0}]

      iex> NxSignal.Shape.to_padding_config({12, 12}, {2, 2}, :same)
      [{0, 1}, {0, 1}]

  ### Error cases

      iex> NxSignal.Shape.to_padding_config({2, 3, 2}, {2, 3, 2}, :foo)
      ** (ArgumentError) invalid padding mode specified, padding must be one of :valid, :same, or a padding configuration, got: :foo

  """
  def to_padding_config(shape, kernel_size, mode) do
    case mode do
      :valid ->
        List.duplicate({0, 0}, tuple_size(shape))

      :same ->
        Enum.zip_with(Tuple.to_list(shape), Tuple.to_list(kernel_size), fn dim, k ->
          padding_size = max(dim - 1 + k - dim, 0)
          {floor(padding_size / 2), ceil(padding_size / 2)}
        end)

      config when is_list(config) ->
        Enum.each(config, fn
          {x, y} when is_integer(x) and is_integer(y) ->
            :ok

          _other ->
            raise ArgumentError,
                  "padding must be a list of {high, low} tuples, where each element is an integer. " <>
                    "Got: #{inspect(config)}"
        end)

        config

      mode ->
        raise ArgumentError,
              "invalid padding mode specified, padding must be one" <>
                " of :valid, :same, or a padding configuration, got:" <>
                " #{inspect(mode)}"
    end
  end
end
