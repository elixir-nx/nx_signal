defmodule NxSignal.InternalTest do
  use NxSignal.Case, async: true, validate_doc_metadata: false
  import NxSignal.Helpers

  describe "lambert_w/3" do
    test "scipy values" do
      for {a, b, y} <- [
            {0, 0, 0},
            {Complex.new(0, 0), 0, 0},
            {Nx.Constants.infinity(), 0, Complex.new(:infinity, 0)},
            {0, -1, Complex.new(:neg_infinity, 0)},
            {0, 1, Complex.new(:neg_infinity, 0)},
            {0, 3, Complex.new(:neg_infinity, 0)},
            {Nx.to_number(Nx.Constants.e()), 0, 1},
            {1, 0, 0.567143290409783873},
            {-Nx.to_number(Nx.Constants.pi()) / 2, 0,
             Complex.new(0, Nx.to_number(Nx.Constants.pi()) / 2)},
            {-Nx.to_number(Nx.log(2)) / 2, 0, -Nx.to_number(Nx.log(2))},
            {0.25, 0, 0.203888354702240164},
            {-0.25, 0, -0.357402956181388903},
            {-1.0 / 10000, 0, -0.000100010001500266719},
            {-0.25, -1, -2.15329236411034965},
            {0.25, -1, Complex.new(-3.00899800997004620, -4.07652978899159763)},
            {-0.25, -1, -2.15329236411034965},
            {0.25, 1, Complex.new(-3.00899800997004620, 4.07652978899159763)},
            {-0.25, 1, Complex.new(-3.48973228422959210, 7.41405453009603664)}
          ] do
        x = NxSignal.Internal.lambert_w(a, b)
        assert_all_close(x, y, atol: 1.0e-13, rtol: 1.0e-10)
      end
    end
  end
end
