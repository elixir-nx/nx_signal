defmodule NxSignal.Convolution do
  @moduledoc """
  Convolution functions
  """

  import Nx.Defn
  import NxSignal.Transforms

  deftransform convolve(in1, in2, opts \\ []) do
    mode = Keyword.get(opts, :mode, "full")
    method = Keyword.get(opts, :method, "direct")

    input_rank =
      case {Nx.rank(in1), Nx.rank(in2)} do
        {0, r} ->
          r

        {r, 0} ->
          r

        {r, r} ->
          r

        {r1, r2} ->
          raise ArgumentError,
                "NxSignal.convolve/3 requires both inputs to have the same rank or one of them to be a scalar, got #{r1} and #{r2}"
      end

    kernel = Nx.reverse(in2)

    kernel_shape =
      case Nx.shape(kernel) do
        {} -> {1, 1, 1, 1}
        {n} -> {1, 1, 1, n}
        shape -> List.to_tuple([1, 1 | Tuple.to_list(shape)])
      end

    kernel = Nx.reshape(kernel, kernel_shape)

    volume_shape =
      case Nx.shape(in1) do
        {} -> {1, 1, 1, 1}
        {n} -> {1, 1, 1, n}
        shape -> List.to_tuple([1, 1 | Tuple.to_list(shape)])
      end

    volume = Nx.reshape(in1, volume_shape)

    opts =
      case mode do
        "same" ->
          kernel_spatial_shape =
            Nx.shape(kernel)
            |> Tuple.to_list()
            |> Enum.drop(2)

          padding =
            Enum.map(kernel_spatial_shape, fn k ->
              pad_total = k - 1
              # integer division for right side
              pad_right = div(pad_total, 2)
              # put the extra padding on the left
              pad_left = pad_total - pad_right
              {pad_left, pad_right}
            end)

          [padding: padding]

        "full" ->
          kernel_spatial_shape =
            Nx.shape(kernel)
            |> Tuple.to_list()
            |> Enum.drop(2)

          padding =
            Enum.map(kernel_spatial_shape, fn k ->
              {k - 1, k - 1}
            end)

          [padding: padding]
      end

    out =
      case method do
        "direct" ->
          Nx.conv(volume, kernel, opts)

        "fft" ->
          fftconvolve(volume, kernel, opts)
      end

    squeeze_axes =
      case input_rank do
        0 ->
          [0, 1, 2, 3]

        1 ->
          [0, 1, 2]

        _ ->
          [0, 1]
      end

    Nx.squeeze(out, axes: squeeze_axes)
  end

  def fftconvolve(volume, kernel, opts \\ []) do
    case {Nx.rank(volume), Nx.rank(kernel)} do
      {1, 1} ->
        Nx.product(volume, kernel)

      {a, b} when a == b ->
        s1 = Nx.shape(volume) |> Tuple.to_list()
        s2 = Nx.shape(kernel) |> Tuple.to_list()

        # Axes initialization
        axes = 0..(Nx.rank(volume) - 1)

        axes =
          for a <- axes, Enum.at(s1, a) != 1 || Enum.at(s2, a) == 1 do
            a
          end

        # Shape setup
        shape =
          for i <- 0..(length(s1) - 1) do
            if i not in axes do
              max(Enum.at(s1, i), Enum.at(s2, i))
            else
              Enum.at(s1, i) + Enum.at(s2, i) - 1
            end
          end

        IO.inspect(shape)

        # Frequency domain conversion
        # IO.inspect([volume, kernel])

        padding_s1 =
          Enum.zip(shape, s1)
          |> Enum.drop(2)
          |> Enum.map(fn {x, y} -> {x - y, 0, 0} end)

        padding_s1 =
          [{0, 0, 0}, {0, 0, 0} | padding_s1]
          |> IO.inspect()

        volume = Nx.pad(volume, 0, padding_s1)

        padding_s2 =
          Enum.zip(shape, s2)
          |> Enum.drop(2)
          |> Enum.map(fn {x, y} -> {x - y, 0, 0} end)

        padding_s2 =
          [{0, 0, 0}, {0, 0, 0} | padding_s2]
          |> IO.inspect()

        kernel = Nx.pad(kernel, 0, padding_s2)

        IO.inspect([s1, s2])

        sp1 = fft_nd(volume, axes: Nx.axes(volume))
        sp2 = fft_nd(kernel, axes: Nx.axes(kernel))
        c = Nx.multiply(sp1, sp2)
        ifft_nd(c, axes: Nx.axes(c))

      _ ->
        raise ArgumentError, message: "Rank of volume and kernel must be equial."
    end
  end
end
