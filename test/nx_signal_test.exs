defmodule NxSignalTest do
  use NxSignal.Case
  doctest NxSignal

  describe "czt/2" do
    test "default params equal DFT" do
      x = Nx.tensor([1.0, 2.0, 3.0, 4.0])
      result = NxSignal.czt(x)
      expected = Nx.fft(Nx.as_type(x, {:c, 64}))

      assert_all_close(result, expected, atol: 1.0e-5)
    end

    test "unit impulse returns all ones" do
      x = Nx.tensor([1.0, 0.0, 0.0, 0.0])
      result = NxSignal.czt(x, output_length: 6)
      ones = Nx.broadcast(Nx.complex(1.0, 0.0), {6})

      assert_all_close(result, ones, atol: 1.0e-5)
    end

    test "M > N is equivalent to zero-padded DFT" do
      x = Nx.tensor([1.0, 2.0, 3.0, 4.0])
      result = NxSignal.czt(x, output_length: 8)
      expected = Nx.fft(Nx.as_type(x, {:c, 64}), length: 8)

      assert_all_close(result, expected, atol: 1.0e-5)
    end

    test "custom contour_start evaluates z-transform off the unit circle" do
      # x = [1, 1], so X(z) = 1 + z^{-1}. At z = 2: X(2) = 1 + 0.5 = 1.5
      x = Nx.tensor([1.0, 1.0])
      a = Nx.complex(2.0, 0.0)
      w = Nx.complex(-1.0, 0.0)
      result = NxSignal.czt(x, output_length: 1, contour_start: a, contour_ratio: w)

      assert_all_close(Nx.real(result[0]), Nx.tensor(1.5, type: {:f, 64}), atol: 1.0e-5)
    end

    test "returns complex output" do
      result = NxSignal.czt(Nx.tensor([1.0, 2.0, 3.0]))

      assert Nx.Type.complex?(Nx.type(result))
    end

    test "preserves f64 precision" do
      x = Nx.tensor([1.0, 2.0, 3.0, 4.0], type: {:f, 64})
      result = NxSignal.czt(x)

      assert Nx.type(result) == {:c, 128}
    end

    test "2D input with axis: 1 matches row-wise 1D CZT" do
      rows = Nx.tensor([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
      result_2d = NxSignal.czt(rows, axis: 1)
      expected = Nx.stack([NxSignal.czt(rows[0]), NxSignal.czt(rows[1])])

      assert_all_close(result_2d, expected, atol: 1.0e-5)
      assert_all_close(NxSignal.czt(rows), result_2d, atol: 1.0e-7)
    end

    test "2D input with axis: 0 matches column-wise 1D CZT" do
      cols = Nx.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
      result_2d = NxSignal.czt(cols, axis: 0)
      col0 = NxSignal.czt(cols[[.., 0]])
      col1 = NxSignal.czt(cols[[.., 1]])
      expected = Nx.stack([col0, col1], axis: 1)

      assert_all_close(result_2d, expected, atol: 1.0e-5)
    end
  end

  describe "zoom_fft/4" do
    test "full range [0, 1] equals DFT" do
      x = Nx.tensor([1.0, 2.0, 3.0, 4.0])
      result = NxSignal.zoom_fft(x, 0.0, 1.0)
      expected = Nx.fft(Nx.as_type(x, {:c, 64}))

      assert_all_close(result, expected, atol: 1.0e-5)
    end

    test "resolves two closely-spaced tones" do
      # Two tones at normalised frequencies 0.49 and 0.51.
      # A 16-point DFT cannot separate them; zoom_fft over [0.45, 0.55] with
      # 64 bins places bin spacing at 0.1/64 ≈ 0.00156, so the two tones land
      # near bins 26 and 38 respectively.
      n = 128
      t = Nx.iota({n}, type: :f64)
      pi2 = 2 * :math.pi()

      x =
        Nx.add(
          Nx.cos(Nx.multiply(t, pi2 * 0.49)),
          Nx.cos(Nx.multiply(t, pi2 * 0.51))
        )

      result = NxSignal.zoom_fft(x, 0.45, 0.55, output_length: 64)
      magnitudes = Nx.abs(result)

      {_vals, peak_indices} = Nx.top_k(magnitudes, k: 2)
      [p1, p2] = peak_indices |> Nx.to_flat_list() |> Enum.sort()

      # tone at 0.49 → bin (0.49 - 0.45) / (0.1 / 64) ≈ 25.6
      # tone at 0.51 → bin (0.51 - 0.45) / (0.1 / 64) ≈ 38.4
      assert abs(p1 - 26) <= 2
      assert abs(p2 - 38) <= 2
    end

    test "output equals equivalent czt call" do
      x = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
      result = NxSignal.zoom_fft(x, 0.1, 0.4, output_length: 16)

      a = Nx.complex(:math.cos(2 * :math.pi() * 0.1), :math.sin(2 * :math.pi() * 0.1))
      w = Nx.complex(:math.cos(-2 * :math.pi() * 0.3 / 16), :math.sin(-2 * :math.pi() * 0.3 / 16))
      expected = NxSignal.czt(x, output_length: 16, contour_ratio: w, contour_start: a)

      assert_all_close(result, expected, atol: 1.0e-5)
    end
  end
end
