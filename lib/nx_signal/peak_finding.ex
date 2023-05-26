defmodule NxSignal.PeakFinding do
  @moduledoc """
  Peak finding algorithms.
  """

  import Nx.Defn
  import Nx, only: [u8: 1, s64: 1]

  defn argrelmin(data, opts \\ []) do
    opts = keyword!(opts, axis: 0, order: 1)
    argrelextrema(data, &Nx.less/2, opts)
  end

  defn argrelmax(data, opts \\ []) do
    opts = keyword!(opts, axis: 0, order: 1)
    argrelextrema(data, &Nx.greater/2, opts)
  end

  defn argrelextrema(data, comparator_fn, opts) do
    data
    |> boolrelextrema(comparator_fn, opts)
    |> nonzero()
  end

  defnp boolrelextrema(data, comparator_fn, opts \\ []) do
    axis = opts[:axis]
    order = opts[:order]
    locs = Nx.iota({Nx.axis_size(data, axis)})

    ones = Nx.broadcast(u8(1), data.shape)
    [ones, _] = Nx.broadcast_vectors([ones, data])

    {results, _} =
      while {results = ones, {data, locs, halt = u8(0), shift = s64(1)}},
            not halt and shift < order + 1 do
        plus = Nx.take(data, Nx.clip(locs + shift, 0, Nx.size(locs) - 1), axis: axis)
        minus = Nx.take(data, Nx.clip(locs - shift, 0, Nx.size(locs) - 1), axis: axis)
        results = comparator_fn.(data, plus) and results
        results = comparator_fn.(data, minus) and results

        {results, {data, locs, not Nx.any(results), shift + 1}}
      end

    results
  end

  defnp nonzero(data) do
    flat_data = Nx.reshape(data, {:auto, 1})

    indices =
      while %{shape: {n, _}} =
              indices = Nx.broadcast(0, {Nx.axis_size(flat_data, 0), Nx.rank(data)}),
            axis <- 0..(Nx.rank(data) - 1) do
        iota = Nx.iota({n, 1}, axis: axis)
        Nx.put_slice(indices, [0, axis], iota)
      end

    Nx.select(flat_data, indices, -1)
  end
end
