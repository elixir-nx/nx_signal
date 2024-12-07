defmodule NxSignal.Convolution do
  @moduledoc """
  Convolution functions
  """

  import Nx.Defn

  deftransform convolve(kernel, in2, mode \\ "full", method \\ "auto") do
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
          nil
          Nx.reshape(kernel, {1, m, n})

        _ ->
          kernel
      end

    in2 =
      case Nx.shape(in2) do
        {m} ->
          Nx.reshape(in2, {1, 1, m})

        {m, n} ->
          nil
          Nx.reshape(in2, {1, m, n})

        _ ->
          in2
      end

    Nx.conv(kernel, in2)
    |> Nx.squeeze()
  end
end
