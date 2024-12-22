defmodule NxSignal.Convolution do
  @moduledoc """
  Convolution functions through various methods.

  Follows the `scipy.signal` conventions.
  """

  import Nx.Defn
  import NxSignal.Transforms

  @doc """
  Computes the convolution of two tensors.

  Given $f[n] \\in \\mathbb{C}^N$ and $k[n] \\in \\mathbb{C}^{K}$, we define the convolution $f * k$ by

  $$
    g[n] = (f * k)[n] = \\sum_{m=0}^{K-1} f[n-m]k[m],
  $$

  where

  $$
  \\tilde{f}[n] =
  \\begin{cases}
    0 & n < 0 \\\\\\\\
    0 & n \\geq N \\\\\\\\
    f[n] & \\text{otherwise}
  \\end{cases}
  $$

  and $\\tilde{k}$ is defined similarly.

  ## Options

    * `:method` - One of `:fft` or `:direct`. Defaults to `:direct`.
    * `:mode` - One of `:full`, `:valid`, or `:same`. Defaults to `:full`.

  ## Examples

    iex> NxSignal.Convolution.convolve(Nx.tensor([1,2,3]), Nx.tensor([3,4,5]))
    #Nx.Tensor<
      f32[5]
      [3.0, 10.0, 22.0, 22.0, 15.0]
    >
  """
  deftransform convolve(in1, in2, opts \\ []) do
    opts = Keyword.validate!(opts, mode: :full, method: :direct)

    if opts[:mode] not in [:full, :same, :valid] do
      raise ArgumentError,
            "expected mode to be one of [:full, :same, :valid], got: #{inspect(opts[:mode])}"
    end

    if opts[:method] not in [:direct, :fft] do
      raise ArgumentError,
            "expected method to be one of [:direct, :fft], got: #{inspect(opts[:method])}"
    end

    case opts[:method] do
      :direct ->
        direct_convolve(in1, in2, opts)

      :fft ->
        fftconvolve(in1, in2, opts)
    end
  end

  @doc """
  Given $f[n] \\in \\mathbb{C}^N$ and $k[n] \\in \\mathbb{C}^{K}$, we define the correlation $f \\star k$ by

   $$
     g(m) = (f * k)[m] = \\sum_{m=0}^{K-1} \\overline{\\tilde{f}[m-n]}\\tilde{k}[m],
   $$

   where

   $$
   \\tilde{f}[n] =
   \\begin{cases}
     0 & n < 0 \\\\\\\\
     0 & n \\geq N \\\\\\\\
     f[n] & \\text{otherwise}
   \\end{cases}
   $$

   and $\\tilde{k}$ is defined similarly.

  ## Options

    * `:method` - One of `:fft` or `:direct`. Defaults to `:direct`.
    * `:mode` - One of `:full`, `:valid`, or `:same`. Defaults to `:full`.

  ## Examples

    iex> NxSignal.Convolution.correlate(Nx.tensor([1,2,3]), Nx.tensor([3,4,5]))
    #Nx.Tensor<
      c64[5]
      [5.0-0.0i, 14.0-0.0i, 26.0-0.0i, 18.0-0.0i, 9.0-0.0i]
    >
  """
  defn correlate(in1, in2, opts \\ []) do
    convolve(in1, Nx.conjugate(Nx.reverse(in2)), opts)
  end

  deftransformp direct_convolve(in1, in2, opts) do
    input_rank =
      case {Nx.rank(in1), Nx.rank(in2)} do
        {0, 0} ->
          0

        {0, r} ->
          raise ArgumentError, message: "Incompatible ranks: {0, #{r}}"

        {r, 0} ->
          raise ArgumentError, message: "Incompatible ranks: {#{r}, 0}"

        {r, r} ->
          r

        {r1, r2} ->
          raise ArgumentError,
                "NxSignal.convolve/3 requires both inputs to have the same rank or one of them to be a scalar, got #{r1} and #{r2}"
      end

    zipped = Enum.zip(Tuple.to_list(Nx.shape(in1)), Tuple.to_list(Nx.shape(in2)))

    ok1 = Enum.all?(for {i, j} <- zipped, do: i >= j)
    ok2 = Enum.all?(for {i, j} <- zipped, do: i <= j)

    {in1, in2} =
      cond do
        opts[:mode] != :valid ->
          {in1, in2}

        ok1 ->
          {in1, in2}

        ok2 ->
          {in2, in1}

        true ->
          raise ArgumentError,
            message:
              "For :valid mode, one must be at least as large as the other in every dimension"
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
        :same ->
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

        :full ->
          kernel_spatial_shape =
            Nx.shape(kernel)
            |> Tuple.to_list()
            |> Enum.drop(2)

          padding =
            Enum.map(kernel_spatial_shape, fn k ->
              {k - 1, k - 1}
            end)

          [padding: padding]

        :valid ->
          [padding: :valid]
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

    out
    |> Nx.squeeze(axes: squeeze_axes)
    |> clip_valid(Nx.shape(volume), Nx.shape(kernel), opts[:mode])
  end

  deftransformp clip_valid(out, in1_shape, in2_shape, :valid) do
    select =
      [in1_shape, in2_shape]
      |> Enum.zip_with(fn [i, j] ->
        0..(i - j)
      end)

    out[select]
  end

  deftransformp clip_valid(out, _, _, _), do: out

  deftransform fftconvolve(in1, in2, opts \\ []) do
    case {Nx.rank(in1), Nx.rank(in2)} do
      {a, b} when a == b ->
        s1 = Nx.shape(in1) |> Tuple.to_list()
        s2 = Nx.shape(in2) |> Tuple.to_list()

        lengths =
          Enum.zip_with(s1, s2, fn ax1, ax2 ->
            ax1 + ax2 - 1
          end)

        axes =
          [s1, s2, Nx.axes(in1)]
          |> Enum.zip_with(fn [ax1, ax2, axis] ->
            if ax1 != 1 and ax2 != 1 do
              axis
            end
          end)
          |> Enum.filter(& &1)

        lengths = Enum.map(axes, &Enum.fetch!(lengths, &1))

        sp1 =
          fft_nd(in1, axes: axes, lengths: lengths)

        sp2 =
          fft_nd(in2, axes: axes, lengths: lengths)

        c = Nx.multiply(sp1, sp2)

        out = ifft_nd(c, axes: axes)

        out =
          if Nx.Type.merge(Nx.type(in1), Nx.type(in2)) |> Nx.Type.complex?() do
            out
          else
            Nx.real(out)
          end

        apply_mode(out, s1, s2, opts[:mode])

      _ ->
        raise ArgumentError, message: "Rank of in1 and in2 must be equal."
    end
  end

  deftransform apply_mode(out, _s1, _s2, :full) do
    out
  end

  deftransform apply_mode(out, s1, _s2, :same) do
    centered(out, s1)
  end

  deftransform apply_mode(out, s1, s2, :valid) do
    {s1, s2} = swap_axes(s1, s2)

    shape_valid =
      for {a, b} <- Enum.zip(s1, s2) do
        a - b + 1
      end

    centered(out, shape_valid)
  end

  deftransformp centered(out, new_shape) do
    start_indices =
      out
      |> Nx.shape()
      |> Tuple.to_list()
      |> Enum.zip_with(new_shape, fn current, new ->
        div(current - new, 2)
      end)

    Nx.slice(out, start_indices, new_shape)
  end

  defp swap_axes(s1, s2) do
    ok1 = Enum.zip_reduce(s1, s2, true, fn a, b, acc -> acc and a >= b end)
    ok2 = Enum.zip_reduce(s2, s1, true, fn a, b, acc -> acc and a >= b end)

    cond do
      ok1 ->
        {s1, s2}

      ok2 ->
        {s2, s1}

      true ->
        raise ArgumentError,
          message:
            "For 'valid' mode, one must be at least as large as the other in every dimension."
    end
  end
end
