defmodule NxSignal.FiltersTest do
  use NxSignal.Case
  doctest NxSignal.Filters

  describe "median/2" do
    test "performs 1D median filter" do
      t = Nx.tensor([10, 9, 8, 7, 1, 4, 5, 3, 2, 6])
      opts = [kernel_shape: {3}]
      expected = Nx.tensor([9.0, 8.0, 7.0, 4.0, 4.0, 4.0, 3.0, 3.0, 3.0, 3.0])

      assert NxSignal.Filters.median(t, opts) == expected
    end

    test "performs 2D median filter" do
      t =
        Nx.tensor([
          [31, 11, 17, 13, 1],
          [1, 3, 19, 23, 29],
          [19, 5, 7, 37, 2]
        ])

      opts = [kernel_shape: {3, 3}]

      expected =
        Nx.tensor([
          [11.0, 13.0, 17.0, 17.0, 17.0],
          [11.0, 13.0, 17.0, 17.0, 17.0],
          [11.0, 13.0, 17.0, 17.0, 17.0]
        ])

      assert NxSignal.Filters.median(t, opts) == expected
    end

    test "performs n-dim median filter" do
      t =
        Nx.tensor([
          [
            [31, 11, 17, 13, 1],
            [1, 3, 19, 23, 29],
            [19, 5, 7, 37, 2]
          ],
          [
            [19, 5, 7, 37, 2],
            [1, 3, 19, 23, 29],
            [31, 11, 17, 13, 1]
          ],
          [
            [1, 3, 19, 23, 29],
            [31, 11, 17, 13, 1],
            [19, 5, 7, 37, 2]
          ]
        ])

      k1 = {3, 3, 1}
      k2 = {3, 3, 3}

      expected1 =
        Nx.tensor([
          [
            [19.0, 5.0, 17.0, 23.0, 2.0],
            [19.0, 5.0, 17.0, 23.0, 2.0],
            [19.0, 5.0, 17.0, 23.0, 2.0]
          ],
          [
            [19.0, 5.0, 17.0, 23.0, 2.0],
            [19.0, 5.0, 17.0, 23.0, 2.0],
            [19.0, 5.0, 17.0, 23.0, 2.0]
          ],
          [
            [19.0, 5.0, 17.0, 23.0, 2.0],
            [19.0, 5.0, 17.0, 23.0, 2.0],
            [19.0, 5.0, 17.0, 23.0, 2.0]
          ]
        ])

      expected2 =
        Nx.tensor([
          [
            [11.0, 13.0, 17.0, 17.0, 17.0],
            [11.0, 13.0, 17.0, 17.0, 17.0],
            [11.0, 13.0, 17.0, 17.0, 17.0]
          ],
          [
            [11.0, 13.0, 17.0, 17.0, 17.0],
            [11.0, 13.0, 17.0, 17.0, 17.0],
            [11.0, 13.0, 17.0, 17.0, 17.0]
          ],
          [
            [11.0, 13.0, 17.0, 17.0, 17.0],
            [11.0, 13.0, 17.0, 17.0, 17.0],
            [11.0, 13.0, 17.0, 17.0, 17.0]
          ]
        ])

      assert NxSignal.Filters.median(t, kernel_shape: k1) == expected1
      assert NxSignal.Filters.median(t, kernel_shape: k2) == expected2
    end

    test "raises if kernel_shape is not compatible" do
      t1 = Nx.iota({10})
      opts1 = [kernel_shape: {5, 5}]

      assert_raise(
        ArgumentError,
        "kernel shape must be of the same rank as the tensor",
        fn -> NxSignal.Filters.median(t1, opts1) end
      )

      t2 = Nx.iota({5, 5})
      opts2 = [kernel_shape: {5, 5, 5}]

      assert_raise(
        ArgumentError,
        "kernel shape must be of the same rank as the tensor",
        fn -> NxSignal.Filters.median(t2, opts2) end
      )
    end
  end

  describe "wiener/2" do
    test "performs n-dim wiener filter with calculated noise" do
      im =
        Nx.tensor(
          [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0, 9.0, 10.0],
            [11.0, 12.0, 13.0, 14.0, 15.0]
          ],
          type: :f64
        )

      kernel_size = {3, 3}

      expected =
        Nx.tensor(
          [
            [
              1.7777777777777777,
              3.0,
              3.6666666666666665,
              4.333333333333333,
              3.111111111111111
            ],
            [4.3366520642506305, 7.0, 8.0, 9.0, 7.58637597408283],
            [
              4.692197051420351,
              7.261706150595039,
              8.748939779474131,
              10.157992415073023,
              9.813815742524799
            ]
          ],
          type: :f64
        )

      assert NxSignal.Filters.wiener(im, kernel_size: kernel_size) == expected
      assert NxSignal.Filters.wiener(im, kernel_size: 3) == expected

      assert NxSignal.Filters.wiener(Nx.as_type(im, :f32), kernel_size: kernel_size) ==
               Nx.tensor([
                 [
                   1.7777777910232544,
                   3.0,
                   3.6666667461395264,
                   4.333333492279053,
                   3.1111111640930176
                 ],
                 [4.3366522789001465, 7.0, 8.0, 9.0, 7.586376190185547],
                 [
                   4.692196846008301,
                   7.261706352233887,
                   8.748939514160156,
                   10.157992362976074,
                   9.81381607055664
                 ]
               ])
    end

    test "performs n-dim wiener filter with parameterized noise" do
      im =
        Nx.tensor(
          [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0, 9.0, 10.0],
            [11.0, 12.0, 13.0, 14.0, 15.0]
          ],
          type: :f64
        )

      kernel_size = {3, 3}

      assert NxSignal.Filters.wiener(im, kernel_size: kernel_size, noise: 10) ==
               Nx.tensor(
                 [
                   [
                     1.7777777777777777,
                     3.0,
                     3.5882352941176467,
                     4.238095238095238,
                     3.7397034596375622
                   ],
                   [5.193548387096774, 7.0, 8.0, 9.0, 8.829787234042554],
                   [
                     7.941747572815534,
                     9.702702702702702,
                     10.938931297709924,
                     12.137254901960784,
                     12.485549132947977
                   ]
                 ],
                 type: :f64
               )

      assert NxSignal.Filters.wiener(Nx.as_type(im, :f32), kernel_size: kernel_size, noise: 10) ==
               Nx.tensor([
                 [
                   1.7777777910232544,
                   3.0,
                   3.588235378265381,
                   4.238095283508301,
                   3.739703416824341
                 ],
                 [5.193548202514648, 7.0, 8.0, 9.0, 8.829787254333496],
                 [
                   7.941747665405273,
                   9.702702522277832,
                   10.938931465148926,
                   12.13725471496582,
                   12.485548973083496
                 ]
               ])

      assert NxSignal.Filters.wiener(im, kernel_size: kernel_size, noise: 0) ==
               Nx.tensor(
                 [
                   [1.0, 2.0, 3.0, 4.0, 5.0],
                   [6.0, 7.0, 8.0, 9.0, 10.0],
                   [11.0, 12.0, 13.0, 14.0, 15.0]
                 ],
                 type: :f64
               )
    end
  end

  describe "firwin/3" do
    # Reference values generated with scipy.signal.firwin

    test "lowpass with default hamming window" do
      expected =
        Nx.tensor([
          0.020103708268285354,
          0.23086668180542194,
          0.4980592198525855,
          0.23086668180542194,
          0.020103708268285354
        ])

      assert_all_close(NxSignal.Filters.firwin(5, [0.3]), expected, atol: 1.0e-5)
    end

    test "highpass with hamming window" do
      expected =
        Nx.tensor([
          0.004998140998601554,
          -0.02905169455437149,
          -0.23351680322070983,
          0.6010660646645265,
          -0.2335168032207099,
          -0.02905169455437152,
          0.004998140998601554
        ])

      assert_all_close(
        NxSignal.Filters.firwin(7, [0.4], pass_zero: false),
        expected,
        atol: 1.0e-5
      )
    end

    test "bandpass with hann window" do
      expected =
        Nx.tensor([
          0.0,
          -0.034265228115753485,
          -0.17548320982592003,
          0.14143709641554006,
          0.5732069654682745,
          0.14143709641554006,
          -0.17548320982592003,
          -0.034265228115753485,
          0.0
        ])

      assert_all_close(
        NxSignal.Filters.firwin(9, [0.2, 0.6], pass_zero: false, window: :hann),
        expected,
        atol: 1.0e-5
      )
    end

    test "bandstop with blackman window" do
      expected =
        Nx.tensor([
          0.0,
          -0.004174601858029537,
          0.0,
          0.17126025417159732,
          0.0,
          0.6658286953728643,
          0.0,
          0.17126025417159732,
          0.0,
          -0.004174601858029537,
          0.0
        ])

      assert_all_close(
        NxSignal.Filters.firwin(11, [0.3, 0.7], window: :blackman),
        expected,
        atol: 1.0e-5
      )
    end

    test "lowpass with kaiser window" do
      expected =
        Nx.tensor([
          -0.003951274147023466,
          0.0,
          0.25034887446528337,
          0.5072047993634803,
          0.25034887446528337,
          0.0,
          -0.003951274147023466
        ])

      assert_all_close(
        NxSignal.Filters.firwin(7, [0.5], window: {:kaiser, 5.0}),
        expected,
        atol: 1.0e-3
      )
    end

    test "lowpass with rectangular window" do
      expected =
        Nx.tensor([
          -0.058404528708691714,
          0.08760679306303756,
          0.28350153764274655,
          0.37459239600581506,
          0.28350153764274655,
          0.08760679306303756,
          -0.058404528708691714
        ])

      assert_all_close(
        NxSignal.Filters.firwin(7, [0.4], window: :rectangular),
        expected,
        atol: 1.0e-5
      )
    end

    test "scale: false returns unscaled coefficients" do
      expected =
        Nx.tensor([
          0.012109227658250522,
          0.13905977799613067,
          0.3,
          0.13905977799613067,
          0.012109227658250522
        ])

      assert_all_close(
        NxSignal.Filters.firwin(5, [0.3], scale: false),
        expected,
        atol: 1.0e-5
      )
    end

    test "cutoff normalised by sampling_rate" do
      expected =
        Nx.tensor([
          0.024553834015016568,
          0.23438946423798604,
          0.48211340349399473,
          0.23438946423798604,
          0.024553834015016568
        ])

      assert_all_close(
        NxSignal.Filters.firwin(5, [1000], sampling_rate: 8000),
        expected,
        atol: 1.0e-5
      )
    end

    test "raises when cutoff is at or above Nyquist" do
      assert_raise ArgumentError, ~r/cutoff must be strictly between 0 and Nyquist/, fn ->
        NxSignal.Filters.firwin(5, [1.0])
      end

      assert_raise ArgumentError, ~r/cutoff must be strictly between 0 and Nyquist/, fn ->
        NxSignal.Filters.firwin(5, [0.0])
      end
    end

    test "raises when even num_taps would produce gain at Nyquist" do
      assert_raise ArgumentError, ~r/odd number of taps/, fn ->
        NxSignal.Filters.firwin(6, [0.4], pass_zero: false)
      end
    end

    test "raises for unknown window" do
      assert_raise ArgumentError, ~r/unknown window/, fn ->
        NxSignal.Filters.firwin(5, [0.3], window: :bogus)
      end
    end
  end
end
