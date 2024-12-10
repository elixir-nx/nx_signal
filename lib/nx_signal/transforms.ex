defmodule NxSignal.Transforms do
  import Nx.Defn

  deftransform fft_nd(tensor, opts \\ []) do
    Enum.reduce(0..(Nx.rank(tensor) - 1), tensor, fn el, acc ->
      if el in Keyword.get(opts, :ignore, []) do
        acc
      else
        Nx.fft(acc, axis: el)
      end
    end)
  end

  deftransform ifft_nd(tensor, opts \\ []) do
    Enum.reduce(0..(Nx.rank(tensor) - 1), tensor, fn el, acc ->
      if el in Keyword.get(opts, :ignore, []) do
        acc
      else
        Nx.ifft(acc, axis: el)
      end
    end)
  end
end
