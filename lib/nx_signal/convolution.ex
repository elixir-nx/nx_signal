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

    kernel =
      case Nx.shape(kernel) do
        {m} ->
          Nx.reshape(kernel, {1, 1, m})

        {m, n} ->
          Nx.reshape(kernel, {1, m, n})

        _ ->
          kernel
      end

    volume =
      case Nx.shape(volume) do
        {m} ->
          Nx.reshape(volume, {1, 1, m})

        {m, n} ->
          Nx.reshape(volume, {1, m, n})

        _ ->
          volume
      end

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
    |> Nx.squeeze()
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
