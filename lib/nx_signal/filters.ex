defmodule NxSignal.Filters do
  @moduledoc """
  Common filter functions.
  """
  import Nx.Defn

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
    |> Enum.map(&(Nx.iota(t.shape, axis: &1)))
    |> Nx.stack(axis: -1)
    |> Nx.reshape({:auto, length(Nx.axes(t))})
  end

  deftransformp start_indices(t, idx_tensor) do
    t
    |> Nx.axes()
    |> Enum.map(&(idx_tensor[&1]))
  end

  deftransformp kernel_lengths(kernel_shape), do: Tuple.to_list(kernel_shape)
end
