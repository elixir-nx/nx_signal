# NxSignal

DSP (Digital Signal Processing) with [Nx](https://github.com/elixir-nx/nx)

## Why NxSignal?

This library comes from the author's urge to experiment with audio processing in Elixir through Nx.
However, the scope is not limited to audio signals. This library aims to provide the tooling for
a more classical approach to dealing with time series, through Fourier Transforms, FIR filters,
IIR filters and similar mathematical tools.

## Getting Started

In order to use `NxSignal`, you need Elixir installed. Then, you can add `NxSignal` as a dependency
to your Mix project:

```elixir
def deps do
  [
    {:nx_signal, "~> 0.1"}
  ]
end
```

You can also use `Mix.install` for standalone development:

```elixir
Mix.install([
  {:nx_signal, "~> 0.1"}
])
```

By default, `NxSignal` only depends directly on `Nx` itself. If you wish to use separate backends
such as `Torchx` or `EXLA`, you need to explicitly depend on them.

All of `NxSignal`'s functionality is provided through `Nx.Defn`, so things should work out of the
box with different backends and compilers.

## Guides (Livebook)

Check out the "guides" directory in the repository for examples. 

## Contributing

Contributions are more than welcome!

Firstly, please make sure you check the issues tracker and the pull requests list for
a similar feature or bugfix to what you wish to contribute.
If there aren't any mentions to be found, open up an issue so that we can discuss the
feature beforehand.
