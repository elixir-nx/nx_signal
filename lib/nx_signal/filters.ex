defmodule NxSignal.Filters do
  @moduledoc """
  Common filter functions.
  """
  import Nx.Defn

  @doc ~S"""
  Performs a median filter on a rank 1 or rank 2 tensor.

  ## Options

    * `:kernel_shape` - the shape of the sliding window.
    It must be compatible with the shape of the tensor.
  """
  @doc type: :filters
  deftransform median(t = %Nx.Tensor{shape: {length}}, opts) do
    validate_median_opts!(t, opts)
    {kernel_length} = opts[:kernel_shape]

    median(Nx.reshape(t, {1, length}), kernel_shape: {1, kernel_length})
    |> Nx.squeeze()
  end

  deftransform median(t = %Nx.Tensor{shape: {_h, _w}}, opts) do
    validate_median_opts!(t, opts)
    median_n(t, opts)
  end

  deftransform median(_t, _opts),
    do: raise(ArgumentError, message: "tensor must be of rank 1 or 2")

  defn median_n(t, opts) do
    {k0, k1} = opts[:kernel_shape]

    idx =
      Nx.stack([Nx.iota(t.shape, axis: 0), Nx.iota(t.shape, axis: 1)], axis: -1)
      |> Nx.reshape({:auto, 2})
      |> Nx.vectorize(:elements)

    t
    |> Nx.slice([idx[0], idx[1]], [k0, k1])
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
end
