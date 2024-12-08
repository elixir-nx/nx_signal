defmodule NxSignal.Convolution do
  @moduledoc """
  Convolution functions
  """

  import Nx.Defn

  deftransform convolve(in1, in2, mode \\ "full", method \\ "auto") do
    kernel = in2
    volume = in1

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
      if mode == "same" do
        [padding: :same]
      else
        []
      end

    Nx.conv(volume, kernel, opts)
    |> Nx.squeeze()
  end
end
