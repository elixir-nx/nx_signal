defmodule NxSignal.Filters do
  @moduledoc """
  Common filter functions.
  """
  import Nx.Defn

  @pi :math.pi()

  @doc ~S"""
  Calculates the normalized sinc function $sinc(t) = \frac{sin(\pi t)}{\pi t}$

  ## Examples

      iex> NxSignal.Filters.sinc(Nx.tensor([0, 0.25, 1]))
      #Nx.Tensor<
        f32[3]
        [1.0, 0.9003162980079651, -2.7827534054836178e-8]
      >
  """
  @doc type: :filters
  defn sinc(t) do
    t = t * @pi
    zero_idx = Nx.equal(t, 0)

    # Define sinc(0) = 1
    Nx.select(zero_idx, 1, Nx.sin(t) / t)
  end

  defn median_filter(t, opts \\ [kernel_size: 3]) do
    {height, width} = Nx.shape(t)
    kernel_tensor = Nx.broadcast(0.0, {opts[:kernel_size]})
    output = Nx.broadcast(0.0, Nx.shape(t))

    {_, _, _, _, _, result} =
      while {i = 0, t, kernel_tensor, height, width, output}, i < height do
        row = Nx.broadcast(0.0, {elem(Nx.shape(t), 1)})

        {_, _, _, _, _, ith_row} =
          while {j = 0, t, i, kernel_tensor, width, row}, j < width do
            ij_median =
              ij_window_median(t, i, j, kernel_tensor)
              |> Nx.broadcast({1})

            {j + 1, t, i, kernel_tensor, width, Nx.put_slice(row, [j], ij_median)}
          end

        {i + 1, t, kernel_tensor, height, width,
         Nx.put_slice(output, [i, 0], Nx.stack(ith_row, axis: 0))}
      end

    result
  end

  defnp ij_window_median(t, i, j, kernel_tensor) do
    {kernel_size} = Nx.shape(kernel_tensor)
    window_pad_size = Nx.round((kernel_size - 1) / 2)

    y_axis_start_idx =
      if i - window_pad_size <= 0 do
        0
      else
        i - window_pad_size
      end
      |> Nx.as_type({:u, 32})

    x_axis_start_idx =
      if j - window_pad_size <= 0 do
        0
      else
        j - window_pad_size
      end
      |> Nx.as_type({:u, 32})

    Nx.slice(
      t,
      [y_axis_start_idx, x_axis_start_idx],
      [kernel_size, kernel_size]
    )
    |> Nx.median()
  end
end
