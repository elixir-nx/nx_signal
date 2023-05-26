defmodule NxSignal.PeakFinding do
  @moduledoc """
  Peak finding algorithms.
  """

  import Nx.Defn
  import Nx, only: [u8: 1, s64: 1]

  @doc """
  Finds a relative minimum along the selected `:axis`.

  A relative minimum is defined by the element being greater
  than its neighbors along the axis `:axis`.

  Returns a map in the following format:

      %{
        indices: #Nx.Tensor<...>,
        valid_indices: #Nx.Tensor<...>
      }

    * `:indices` - the `{n, rank}` tensor of indices.
      Contains `-1` as a placeholder for invalid indices.

    * `:valid_indices` - the number of valid indices that lead the tensor.

  ## Options

    * `:axis` - the axis along which to do comparisons. Defaults to 0.
    * `:order` - the number of neighbor samples considered for the
      comparison in each direction. Defaults to 1.

  ## Examples

      iex> x = Nx.tensor([2, 1, 2, 3, 2, 0, 1, 0])
      iex> %{indices: indices, valid_indices: valid_indices} = NxSignal.PeakFinding.argrelmin(x)
      iex> valid_indices
      #Nx.Tensor<
        u64
        2
      >
      iex> indices
      #Nx.Tensor<
        s64[8][1]
        [
          [1],
          [5],
          [-1],
          [-1],
          [-1],
          [-1],
          [-1],
          [-1]
        ]
      >
      iex> Nx.slice_along_axis(indices, 0, Nx.to_number(valid_indices), axis: 0)
      #Nx.Tensor<
        s64[2][1]
        [
          [1],
          [5]
        ]
      >

  For the same tensor in the previous example, we can use `:order` to check if
  the relative maxima are extrema in a wider neighborhood.

      iex> x = Nx.tensor([2, 1, 2, 3, 2, 0, 1, 0])
      iex> %{indices: indices, valid_indices: valid_indices} = NxSignal.PeakFinding.argrelmin(x, order: 3)
      iex> valid_indices
      #Nx.Tensor<
        u64
        1
      >
      iex> indices
      #Nx.Tensor<
        s64[8][1]
        [
          [1],
          [-1],
          [-1],
          [-1],
          [-1],
          [-1],
          [-1],
          [-1]
        ]
      >
      iex> Nx.slice_along_axis(indices, 0, Nx.to_number(valid_indices), axis: 0)
      #Nx.Tensor<
        s64[1][1]
        [
          [1]
        ]
      >

  We can also apply this function to tensors with a larger rank:

      iex> x = Nx.tensor([[1, 2, 1, 2], [6, 2, 0, 0], [5, 3, 4, 4]])
      iex> %{indices: indices, valid_indices: valid_indices} = NxSignal.PeakFinding.argrelmin(x)
      iex> valid_indices
      #Nx.Tensor<
        u64
        2
      >
      iex> indices[0..1]
      #Nx.Tensor<
        s64[2][2]
        [
          [1, 2],
          [1, 3]
        ]
      >
      iex> %{indices: indices} = NxSignal.PeakFinding.argrelmin(x, axis: 1)
      iex> valid_indices
      #Nx.Tensor<
        u64
        2
      >
      iex> indices[0..1]
      #Nx.Tensor<
        s64[2][2]
        [
          [0, 2],
          [2, 1]
        ]
      >

  """
  @doc type: :peak_finding
  defn argrelmin(data, opts \\ []) do
    opts = keyword!(opts, axis: 0, order: 1)
    argrelextrema(data, &Nx.less/2, opts)
  end

  @doc """
  Finds a relative maximum along the selected `:axis`.

  A relative maximum is defined by the element being greater
  than its neighbors along the axis `:axis`.

  Returns a map in the following format:

      %{
        indices: #Nx.Tensor<...>,
        valid_indices: #Nx.Tensor<...>
      }

    * `:indices` - the `{n, rank}` tensor of indices.
      Contains `-1` as a placeholder for invalid indices.

    * `:valid_indices` - the number of valid indices that lead the tensor.

  ## Options

    * `:axis` - the axis along which to do comparisons. Defaults to 0.
    * `:order` - the number of neighbor samples considered for the
      comparison in each direction. Defaults to 1.

  ## Examples

      iex> x = Nx.tensor([2, 1, 2, 3, 2, 0, 1, 0])
      iex> %{indices: indices, valid_indices: valid_indices} = NxSignal.PeakFinding.argrelmax(x)
      iex> valid_indices
      #Nx.Tensor<
        u64
        2
      >
      iex> indices
      #Nx.Tensor<
        s64[8][1]
        [
          [3],
          [6],
          [-1],
          [-1],
          [-1],
          [-1],
          [-1],
          [-1]
        ]
      >
      iex> Nx.slice_along_axis(indices, 0, Nx.to_number(valid_indices), axis: 0)
      #Nx.Tensor<
        s64[2][1]
        [
          [3],
          [6]
        ]
      >

  For the same tensor in the previous example, we can use `:order` to check if
  the relative maxima are extrema in a wider neighborhood.

      iex> x = Nx.tensor([2, 1, 2, 3, 2, 0, 1, 0])
      iex> %{indices: indices, valid_indices: valid_indices} = NxSignal.PeakFinding.argrelmax(x, order: 3)
      iex> valid_indices
      #Nx.Tensor<
        u64
        1
      >
      iex> indices
      #Nx.Tensor<
        s64[8][1]
        [
          [3],
          [-1],
          [-1],
          [-1],
          [-1],
          [-1],
          [-1],
          [-1]
        ]
      >
      iex> Nx.slice_along_axis(indices, 0, Nx.to_number(valid_indices), axis: 0)
      #Nx.Tensor<
        s64[1][1]
        [
          [3]
        ]
      >

  We can also apply this function to tensors with a larger rank:

      iex> x = Nx.tensor([[1, 2, 1, 2], [6, 2, 0, 0], [5, 3, 4, 4]])
      iex> %{indices: indices, valid_indices: valid_indices} = NxSignal.PeakFinding.argrelmax(x)
      iex> valid_indices
      #Nx.Tensor<
        u64
        1
      >
      iex> indices[0]
      #Nx.Tensor<
        s64[2]
        [1, 0]
      >
      iex> %{indices: indices} = NxSignal.PeakFinding.argrelmax(x, axis: 1)
      iex> valid_indices
      #Nx.Tensor<
        u64
        1
      >
      iex> indices[0]
      #Nx.Tensor<
        s64[2]
        [0, 1]
      >

  """
  @doc type: :peak_finding
  defn argrelmax(data, opts \\ []) do
    opts = keyword!(opts, axis: 0, order: 1)
    argrelextrema(data, &Nx.greater/2, opts)
  end

  @doc """
  Finds a relative extrema along the selected `:axis`.

  A relative extremum is defined by the given `comparator_fn`
  function of arity 2 function that returns a boolean tensor.

  This is the function upon which `&argrelmax/2` and `&argrelmin/2`
  are implemented.

  Returns a map in the following format:

      %{
        indices: #Nx.Tensor<...>,
        valid_indices: #Nx.Tensor<...>
      }

    * `:indices` - the `{n, rank}` tensor of indices.
      Contains `-1` as a placeholder for invalid indices.

    * `:valid_indices` - the number of valid indices that lead the tensor.

  ## Options

    * `:axis` - the axis along which to do comparisons. Defaults to 0.
    * `:order` - the number of neighbor samples considered for the
      comparison in each direction. Defaults to 1.

  ## Examples

  First, do read the examples on `argrelmax/2` keeping in mind that
  it is equivalent to `argrelextrema(&1, &Nx.greater/2, &2)`, as well
  as `argrelmin/2` which is equivalent to `argrelextrema(&1, &Nx.less/2, &2)`.

  Having that in mind, we will expand on those concepts by using a custom function.
  For instance, we can change the definition of a relative maximum to one where
  a number is a relative maximum if it is greater than or equal to the double of its
  neighbors, as follows:

      iex> comparator = fn x, y -> Nx.greater_equal(x, Nx.multiply(y, 2)) end
      iex> x = Nx.tensor([0, 1, 3, 2, 0, 1, 0, 0, 0, 2, 1])
      iex> result = NxSignal.PeakFinding.argrelextrema(x, comparator)
      iex> result.valid_indices
      #Nx.Tensor<
        u64
        3
      >
      iex> result.indices[0..2]
      #Nx.Tensor<
        s64[3][1]
        [
          [5],
          [7],
          [9]
        ]
      >

  Same applies for finding local minima. In the next example, we
  find all local minima (i.e. `&Nx.less/2`) that are
  different to the global minimum.

      iex> x = Nx.tensor([0, 1, 0, 2, 1, 3, 0, 1])
      iex> global_minimum = Nx.reduce_min(x)
      iex> comparator = fn x, y ->
      ...> x_not_global = Nx.not_equal(x, global_minimum)
      ...> y_not_global = Nx.not_equal(y, global_minimum)
      ...> both_not_global = Nx.logical_and(x_not_global, y_not_global)
      ...> Nx.logical_and(Nx.less(x, y), both_not_global)
      ...> end
      iex> result = NxSignal.PeakFinding.argrelextrema(x, comparator)
      iex> result.valid_indices
      #Nx.Tensor<
        u64
        1
      >
      iex> result.indices[0..0]
      #Nx.Tensor<
        s64[1][1]
        [
          [4]
        ]
      >
  """
  @doc type: :peak_finding
  defn argrelextrema(data, comparator_fn, opts \\ []) do
    opts = keyword!(opts, axis: 0, order: 1)

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

  deftransformp nonzero(data) do
    flat_data = Nx.reshape(data, {:auto, 1})

    indices =
      for axis <- 0..(Nx.rank(data) - 1),
          reduce: Nx.broadcast(0, {Nx.axis_size(flat_data, 0), Nx.rank(data)}) do
        %{shape: {n, _}} = indices ->
          iota = data.shape |> Nx.iota(axis: axis) |> Nx.reshape({n, 1})
          Nx.put_slice(indices, [0, axis], iota)
      end

    indices_with_mask =
      Nx.select(
        Nx.broadcast(flat_data, indices.shape),
        indices,
        Nx.broadcast(-1, indices.shape)
      )

    order = Nx.argsort(Nx.squeeze(flat_data, axes: [1]), axis: 0, direction: :desc)

    %{indices: Nx.take(indices_with_mask, order), valid_indices: Nx.sum(flat_data)}
  end
end
