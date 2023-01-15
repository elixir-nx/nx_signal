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

  @doc """
  Dilates the given input shape according to dilation.

  ## Examples

      iex> NxSignal.Shape.dilate({3, 3, 3}, [1, 2, 1])
      {3, 5, 3}

      iex> NxSignal.Shape.dilate({2, 4, 2}, [3, 1, 3])
      {4, 4, 4}
  """
  def dilate(shape, dilation) when is_tuple(shape) and is_list(dilation) do
    unless Enum.all?(dilation, &(&1 >= 1)) do
      raise ArgumentError,
            "dilation rates must be greater than or equal to 1" <>
              " got #{inspect(dilation)}"
    end

    dilated_padding_config = Enum.map(dilation, fn x -> {0, 0, x - 1} end)
    pad(shape, dilated_padding_config)
  end

  @doc """
  Output shape after a pooling or reduce window operation.

  ## Examples

    iex> NxSignal.Shape.pool({3, 3}, {1, 2}, [1, 1], :valid, [1, 1])
    {{3, 2}, [{0, 0}, {0, 0}]}

    iex> NxSignal.Shape.pool({3, 2, 3}, {2, 1, 1}, [1, 2, 1], :same, [1, 1, 1])
    {{3, 1, 3}, [{0, 1}, {0, 0}, {0, 0}]}

  ### Error cases

    iex> NxSignal.Shape.pool({1, 2, 3}, {2, 1, 1}, [1, 1, 1], :valid, [1, 1, 1])
    ** (ArgumentError) window dimensions would result in empty tensor which is not currently supported in Nx, please open an issue if you'd like this behavior to change

    iex> NxSignal.Shape.pool({1, 2, 3}, {2, 1}, [1, 1, 1], :valid, [1, 1, 1])
    ** (ArgumentError) invalid window dimensions, rank of shape (3) does not match rank of window (2)

    iex> NxSignal.Shape.pool({1, 2, 3}, {2, 1, 1}, [1, 1], :valid, [1, 1, 1])
    ** (ArgumentError) invalid stride dimensions, rank of shape (3) does not match rank of stride (2)
  """
  def pool(shape, kernel_size, strides, padding, kernel_dilation) do
    validate_window!(shape, kernel_size)
    validate_strides!(shape, strides)

    kernel_size = dilate(kernel_size, kernel_dilation)

    padding_config = to_padding_config(shape, kernel_size, padding)
    shape = pad(shape, Enum.map(padding_config, fn {x, y} -> {x, y, 0} end))
    {List.to_tuple(do_pool(strides, shape, kernel_size, 0)), padding_config}
  end

  defp do_pool([], _shape, _window, _pos), do: []

  defp do_pool([s | strides], shape, window, pos) do
    dim = elem(shape, pos)
    w = elem(window, pos)
    new_dim = div(dim - w, s) + 1

    if new_dim <= 0 do
      raise ArgumentError,
            "window dimensions would result in empty tensor" <>
              " which is not currently supported in Nx, please" <>
              " open an issue if you'd like this behavior to change"
    end

    [new_dim | do_pool(strides, shape, window, pos + 1)]
  end

  # Ensures the window is valid given the shape.
  # A window is valid as long as it's rank matches
  # the rank of the given shape.
  defp validate_window!(shape, window)

  defp validate_window!(shape, window) when tuple_size(shape) != tuple_size(window),
    do:
      raise(
        ArgumentError,
        "invalid window dimensions, rank of shape (#{tuple_size(shape)})" <>
          " does not match rank of window (#{tuple_size(window)})"
      )

  defp validate_window!(_, _), do: :ok

  # Ensures the strides are valid given the shape.
  # A stride is valid as long as it's rank matches
  # the rank of the given shape.
  defp validate_strides!(shape, strides)

  defp validate_strides!(shape, strides) when tuple_size(shape) != length(strides),
    do:
      raise(
        ArgumentError,
        "invalid stride dimensions, rank of shape (#{tuple_size(shape)})" <>
          " does not match rank of stride (#{length(strides)})"
      )

  defp validate_strides!(_, _), do: :ok

  @doc """
  Output shape after a padding operation.

  ## Examples

      iex> NxSignal.Shape.pad({3, 2, 4}, [{0, 1, 0}, {1, 2, 0}, {1, 1, 0}])
      {4, 5, 6}

      iex> NxSignal.Shape.pad({}, [])
      {}

      iex> NxSignal.Shape.pad({2, 2}, [{1, 1, 0}, {0, 0, 0}])
      {4, 2}

      iex> NxSignal.Shape.pad({2, 3}, [{0, 0, 1}, {0, 0, 1}])
      {3, 5}

  ### Error cases

      iex> NxSignal.Shape.pad({2, 2, 3}, [{0, 1, 0}, {1, 2, 0}])
      ** (ArgumentError) invalid padding configuration, rank of padding configuration and shape must match
  """
  def pad(shape, padding_config) do
    shape
    |> Tuple.to_list()
    |> padded_dims(padding_config, [])
    |> Enum.reverse()
    |> List.to_tuple()
  end

  defp padded_dims([], [], acc), do: acc

  defp padded_dims([_ | _], [], _acc),
    do:
      raise(
        ArgumentError,
        "invalid padding configuration, rank of padding configuration" <>
          " and shape must match"
      )

  defp padded_dims([], [_ | _], _acc),
    do:
      raise(
        ArgumentError,
        "invalid padding configuration, rank of padding configuration" <>
          " and shape must match"
      )

  defp padded_dims([s | shape], [{edge_low, edge_high, interior} | config], acc) do
    interior_padding_factor = (s - 1) * interior
    padded_dims(shape, config, [s + interior_padding_factor + edge_low + edge_high | acc])
  end
end
