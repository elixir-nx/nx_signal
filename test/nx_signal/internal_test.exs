defmodule NxSignal.InternalTest do
  use NxSignal.Case, async: true, validate_doc_metadata: false

  describe "lambert_w/3" do
    test "scipy values" do
      for {a, b, y} <- [
            {0, 0, 0},
            {Complex.new(0, 0), 0, 0},
            {Nx.Constants.infinity(:f64), 0, Complex.new(:infinity, 0)},
            {0, -1, Complex.new(:neg_infinity, 0)},
            {0, 1, Complex.new(:neg_infinity, 0)},
            {0, 3, Complex.new(:neg_infinity, 0)},
            {Nx.to_number(Nx.Constants.e({:f, 64})), 0, 1},
            {1, 0, 0.567143290409783873},
            {-Nx.to_number(Nx.Constants.pi(:f64)) / 2, 0,
             Complex.new(0, Nx.to_number(Nx.Constants.pi(:f64)) / 2)},
            {-:math.log(2.0) / 2, 0, -:math.log(2)},
            {0.25, 0, 0.203888354702240164},
            {-0.25, 0, -0.357402956181388903},
            {-1.0 / 10000, 0, -0.000100010001500266719},
            {-0.25, -1, -2.15329236411034965},
            {0.25, -1, Complex.new(-3.00899800997004620, -4.07652978899159763)},
            {-0.25, -1, -2.15329236411034965},
            {0.25, 1, Complex.new(-3.00899800997004620, 4.07652978899159763)},
            {-0.25, 1, Complex.new(-3.48973228422959210, 7.41405453009603664)},
            {-4, 0, Complex.new(0.67881197132094523, 1.91195078174339937)},
            {-4, 1, Complex.new(-0.6674310712980098, 7.76827456802783084)},
            {-4, -1, Complex.new(0.67881197132094523, -1.91195078174339937)},
            {1000, 0, 5.24960285240159623},
            {1000, 1, Complex.new(4.91492239981054535, 5.44652615979447070)},
            {1000, -1, Complex.new(4.91492239981054535, -5.44652615979447070)},
            {1000, 5, Complex.new(3.5010625305312892, 29.9614548941181328)},
            {Complex.new(3, 4), 0, Complex.new(1.281561806123775878, 0.533095222020971071)},
            {Complex.new(-0.4, 0.4), 0, Complex.new(-0.10396515323290657, 0.61899273315171632)},
            {Complex.new(3, 4), 1, Complex.new(-0.11691092896595324, 5.61888039871282334)},
            {Complex.new(3, 4), -1, Complex.new(0.25856740686699742, -3.85211668616143559)},
            {-0.5, -1, Complex.new(-0.794023632344689368, -0.770111750510379110)},
            {-1.0 / 10000, 1, Complex.new(-11.82350837248724344, 6.80546081842002101)},
            {-1.0 / 10000, -1, -11.6671145325663544},
            {-1.0 / 10000, -2, Complex.new(-11.82350837248724344, -6.80546081842002101)},
            {-1.0 / 100_000, 4, Complex.new(-14.9186890769540539, 26.1856750178782046)},
            {-1.0 / 100_000, 5, Complex.new(-15.0931437726379218666, 32.5525721210262290086)},
            {Complex.divide(Complex.new(2, 1), 10), 0,
             Complex.new(0.173704503762911669, 0.071781336752835511)},
            {Complex.divide(Complex.new(2, 1), 10), 1,
             Complex.new(-3.21746028349820063, 4.56175438896292539)},
            {Complex.divide(Complex.new(2, 1), 10), -1,
             Complex.new(-3.03781405002993088, -3.53946629633505737)},
            {Complex.divide(Complex.new(2, 1), 10), 4,
             Complex.new(-4.6878509692773249, 23.8313630697683291)},
            {Complex.divide(Complex.new(-2, -1), 10), 0,
             Complex.new(-0.226933772515757933, -0.164986470020154580)},
            {Complex.divide(Complex.new(-2, -1), 10), 1,
             Complex.new(-2.43569517046110001, 0.76974067544756289)},
            {Complex.divide(Complex.new(-2, -1), 10), -1,
             Complex.new(-3.54858738151989450, -6.91627921869943589)},
            {Complex.divide(Complex.new(-2, -1), 10), 4,
             Complex.new(-4.5500846928118151, 20.6672982215434637)},
            {Nx.Constants.pi(:f64), 0, 1.073658194796149172092178407024821347547745350410314531}
          ] do
        x = NxSignal.Internal.lambert_w(a, b)
        assert_all_close(x, as_tensor(y), atol: 1.0e-13, rtol: 1.0e-10)
      end
    end
  end

  defp as_tensor(a) when is_struct(a, Complex) do
    Nx.tensor(a, type: :c128)
  end

  defp as_tensor(a) when is_struct(a, Nx.Tensor) do
    a
  end

  defp as_tensor(a) do
    Nx.f64(a)
  end
end
