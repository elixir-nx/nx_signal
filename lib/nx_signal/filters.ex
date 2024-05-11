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
  def median(t = %Nx.Tensor{shape: {length}}, opts) do
    validate_median_opts!(t, opts)
    {kernel_length} = opts[:kernel_shape]
    median(Nx.reshape(t, {1, length}), kernel_shape: {1, kernel_length})
  end

  def median(t = %Nx.Tensor{shape: {_h, _w}}, opts) do
    validate_median_opts!(t, opts)
    median_n(t, opts)
  end

  def median(_t, _opts), do: raise(ArgumentError, message: "tensor must be of rank 1 or 2")

  defnp median_n(t, opts) do
    kernel_shape = opts[:kernel_shape]

    {height, width} = Nx.shape(t)

    kernel_tensor = Nx.broadcast(0.0, kernel_shape)
    output = Nx.broadcast(0.0, Nx.shape(t))

    {result, _} =
      while {output, {i = 0, t, kernel_tensor, height, width}}, i < height do
        row = Nx.broadcast(0.0, {elem(Nx.shape(t), 1)})

        {ith_row, _} =
          while {row, {j = 0, t, i, kernel_tensor, width}}, j < width do
            median =
              window_median(t, i, j, kernel_tensor)
              |> Nx.broadcast({1})

            {Nx.put_slice(row, [j], median), {j + 1, t, i, kernel_tensor, width}}
          end

        {Nx.put_slice(output, [i, 0], Nx.stack(ith_row, axis: 0)),
         {i + 1, t, kernel_tensor, height, width}}
      end

    Nx.squeeze(result)
  end

  defnp window_median(t, i, j, kernel_tensor) do
    kernel_shape = Nx.shape(kernel_tensor)
    kernel_dims = [elem(kernel_shape, 0), elem(kernel_shape, 1)]

    padding_y = Nx.round((elem(kernel_shape, 0) - 1) / 2)
    padding_x = Nx.round((elem(kernel_shape, 1) - 1) / 2)

    y_axis_start_idx =
      if i - padding_y <= 0 do
        0
      else
        i - padding_y
      end
      |> Nx.as_type({:u, 32})

    x_axis_start_idx =
      if j - padding_x <= 0 do
        0
      else
        j - padding_x
      end
      |> Nx.as_type({:u, 32})

    Nx.slice(
      t,
      [y_axis_start_idx, x_axis_start_idx],
      kernel_dims
    )
    |> Nx.median()
  end

  defn validate_median_opts!(t, opts) do
    keyword!(opts, [:kernel_shape])

    if Nx.rank(t) != Nx.rank(opts[:kernel_shape]) do
      raise ArgumentError, message: "kernel shape must be of the same rank as the tensor"
    end
  end
end
