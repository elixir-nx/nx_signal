defmodule NxSignal.Filters do
  @moduledoc """
  Common filter functions.
  """
  import Nx.Defn
  import NxSignal.Convolution

  @doc ~S"""
  Performs a median filter on a tensor.

  ## Options

    * `:kernel_shape` - the shape of the sliding window.
    It must be compatible with the shape of the tensor.
  """
  @doc type: :filters
  defn median(t, opts) do
    validate_median_opts!(t, opts)

    idx =
      t
      |> idx_tensor()
      |> Nx.vectorize(:elements)

    t
    |> Nx.slice(start_indices(t, idx), kernel_lengths(opts[:kernel_shape]))
    |> Nx.median()
    |> Nx.devectorize(keep_names: false)
    |> Nx.reshape(t.shape)
    |> Nx.as_type({:f, 32})
  end

  deftransformp validate_median_opts!(t, opts) do
    Keyword.validate!(opts, [:kernel_shape])

    if Nx.rank(t) != Nx.rank(opts[:kernel_shape]) do
      raise ArgumentError, message: "kernel shape must be of the same rank as the tensor"
    end
  end

  deftransformp idx_tensor(t) do
    t
    |> Nx.axes()
    |> Enum.map(&Nx.iota(t.shape, axis: &1))
    |> Nx.stack(axis: -1)
    |> Nx.reshape({:auto, length(Nx.axes(t))})
  end

  deftransformp start_indices(t, idx_tensor) do
    t
    |> Nx.axes()
    |> Enum.map(&idx_tensor[&1])
  end

  deftransformp kernel_lengths(kernel_shape), do: Tuple.to_list(kernel_shape)

  @doc """
  Applies a Wiener filter to the given Nx tensor.

  ## Options
      * `:kernel_size` - filter size (scalar or tuple). Defaults to `3`.
      * `:noise` - noise power, auto-estimated if `nil`. Defaults to `nil`.

  ## Examples

      iex> t = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
      iex> NxSignal.Filters.wiener(t, kernel_size: {2, 2}, noise: 10)
      #Nx.Tensor<
        f32[3][3]
        [
          [0.25, 0.75, 1.25],
          [1.25, 3.0, 4.0],
          [2.75, 6.0, 7.0]
        ]
      >
  """
  @doc type: :filters
  deftransform wiener(t, opts \\ []) do
    # Validate and extract options
    opts = Keyword.validate!(opts, noise: nil, kernel_size: 3)

    rank = Nx.rank(t)
    kernel_size = Keyword.fetch!(opts, :kernel_size)
    noise = Keyword.fetch!(opts, :noise)

    # Ensure `kernel_size` is a tuple
    kernel_size =
      cond do
        is_integer(kernel_size) -> Tuple.duplicate(kernel_size, rank)
        is_tuple(kernel_size) -> kernel_size
        true -> raise ArgumentError, "kernel_size must be an integer or tuple"
      end

    # Convert `nil` noise to `0.0` so it's always a valid tensor
    noise_t = if is_nil(noise), do: Nx.tensor(0.0), else: Nx.tensor(noise)

    # Compute filter window size
    size = Tuple.to_list(kernel_size) |> Enum.reduce(1, &*/2)

    # Ensure the kernel is the same size as the filter window
    kernel = Nx.broadcast(1.0, kernel_size)

    t
    |> Nx.as_type(:f64)
    |> wiener_n(kernel, noise_t, calculate_noise: is_nil(noise), size: size)
    |> Nx.as_type(Nx.type(t))
  end

  defnp wiener_n(t, kernel, noise, opts) do
    size = opts[:size]

    # Compute local mean using "same" mode in correlation
    l_mean = correlate(t, kernel, mode: :same) / size

    # Compute local variance
    l_var =
      correlate(t ** 2, kernel, mode: :same)
      |> Nx.divide(size)
      |> Nx.subtract(l_mean ** 2)

    # Ensure `noise` is a tensor to avoid `nil` issues in `defnp`
    noise =
      case opts[:calculate_noise] do
        true -> Nx.mean(l_var)
        false -> noise
      end

    # Apply Wiener filter formula
    res = (t - l_mean) * (1 - noise / l_var)
    Nx.select(l_var < noise, l_mean, res + l_mean)
  end
end
