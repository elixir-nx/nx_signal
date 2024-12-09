defmodule NxSignal.Convolution do
  @moduledoc """
  Convolution functions
  """

  import Nx.Defn

  deftransform convolve(in1, in2, opts \\ []) do
    kernel = in2
    volume = in1

    mode = Keyword.get(opts, :mode, "full")

    axes =
      Nx.axes(kernel)

    kernel =
      if Nx.Type.complex?(Nx.type(kernel)) do
        Nx.conjugate(Nx.reverse(kernel, axes: axes))
      else
        Nx.reverse(kernel, axes: axes)
      end

    kernel_shape =
      Nx.shape(kernel)
      |> Tuple.insert_at(0, 1)
      |> Tuple.insert_at(0, 1)

    kernel = Nx.reshape(kernel, kernel_shape)

    volume_shape =
      Nx.shape(volume)
      |> Tuple.insert_at(0, 1)
      |> Tuple.insert_at(0, 1)

    volume = Nx.reshape(volume, volume_shape)

    opts =
      case mode do
        "same" ->
          [padding: :same]

        "full" ->
          padding =
            Nx.shape(volume)
            |> Tuple.to_list()
            |> Enum.slice(2..-1)
            |> Enum.map(&(&1 - 1))
            |> Enum.map(&{&1, &1})

          [padding: padding]
      end

    Nx.conv(kernel, volume, opts)
    |> Nx.reverse()
    |> shape_output(mode, Nx.shape(volume))
    |> then(fn x -> Nx.reshape(x, drop_first_two(Nx.shape(x))) end)
  end

  defp drop_first_two(shape) do
    shape
    |> Tuple.to_list()
    |> Enum.slice(2..-1)
    |> List.to_tuple()
  end

  defp shape_output(out, "full", _shape) do
    out
  end

  defp shape_output(out, "same", shape) do
    ac =
      shape
      |> Tuple.to_list()
      |> Enum.map(&(0..(&1 - 1)))

    out[ac]
  end
end
