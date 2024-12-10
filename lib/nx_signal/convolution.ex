defmodule NxSignal.Convolution do
  @moduledoc """
  Convolution functions
  """

  import Nx.Defn
  import NxSignal.Transforms

  deftransform convolve(in1, in2, opts \\ []) do
    opts = Keyword.validate!(opts, mode: "full", method: "direct")

    case opts[:method] do
      "direct" ->
        direct_convolve(in1, in2, opts)

      "fft" ->
        fftconvolve(in1, in2, opts)
    end
  end

  defp direct_convolve(in1, in2, opts) do
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
      case opts[:mode] do
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

    out = Nx.conv(volume, kernel, opts)

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

  deftransform fftconvolve(in1, in2, opts \\ []) do
    case {Nx.rank(in1), Nx.rank(in2)} do
      {1, 1} ->
        Nx.product(in1, in2)

      {a, b} when a == b ->
        s1 = Nx.shape(in1) |> Tuple.to_list()
        s2 = Nx.shape(in2) |> Tuple.to_list()

        lengths =
          Enum.zip_with(s1, s2, fn ax1, ax2 ->
            case opts[:mode] do
              "full" -> ax1 + ax2 - 1
              "same" -> ax1
            end
          end)

        dbg(lengths)

        sp1 =
          fft_nd(in1, axes: Nx.axes(in1), lengths: lengths)

        sp2 =
          fft_nd(in2, axes: Nx.axes(in2), lengths: lengths)

        dbg({sp1, sp2})

        c = Nx.multiply(sp1, sp2)

        ifft_nd(c, axes: Nx.axes(c))

      _ ->
        raise ArgumentError, message: "Rank of in1 and in2 must be equal."
    end
  end
end
