defmodule NxSignal.Transforms do
  import Nx.Defn

  deftransform fft_nd(tensor, opts \\ []) do
    Enum.reduce(
      Enum.zip(Keyword.get(opts, :axes, [-1]), Keyword.get(opts, :lengths, [])),
      tensor,
      fn {axis, len}, acc ->
        Nx.fft(acc, axis: axis, length: len)
      end
    )
  end

  deftransform ifft_nd(tensor, opts \\ []) do
    Enum.reduce(
      Enum.zip(Keyword.get(opts, :axes, [-1]), Keyword.get(opts, :lengths, [])),
      tensor,
      fn {axis, len}, acc ->
        Nx.ifft(acc, axis: axis, length: len)
      end
    )
  end
end
