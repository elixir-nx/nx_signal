defmodule NxSignal.Case do
  use ExUnit.CaseTemplate
  import ExUnit.Assertions

  using opts do
    validate_doc_metadata = Keyword.get(opts, :validate_doc_metadata, true)

    quote do
      import NxSignal.Case

      if unquote(validate_doc_metadata) do
        test "defines doc :type" do
          validate_doc_metadata(__MODULE__)
        end
      end
    end
  end

  @doctypes [
    :time_frequency,
    :windowing,
    :filters,
    :waveforms,
    :peak_finding
  ]

  def validate_doc_metadata(module) do
    [h | t] = module |> Module.split() |> Enum.reverse()
    h = String.trim_trailing(h, "Test")
    mod = [h | t] |> Enum.reverse() |> Module.concat()

    {:docs_v1, _, :elixir, "text/markdown", _docs, _metadata, entries} = Code.fetch_docs(mod)

    for {{:function, name, arity}, _ann, _signature, docs, metadata} <- entries,
        is_map(docs) and map_size(docs) > 0,
        metadata[:type] not in @doctypes do
      flunk("invalid @doc type: #{inspect(metadata[:type])} for #{name}/#{arity}")
    end
  end

  @doc """
  Asserts `lhs` is close to `rhs`.
  """
  def assert_all_close(lhs, rhs, opts \\ []) do
    atol = opts[:atol] || 1.0e-4
    rtol = opts[:rtol] || 1.0e-4

    if Nx.all_close(lhs, rhs, atol: atol, rtol: rtol, equal_nan: opts[:equal_nan]) !=
         Nx.tensor(1, type: {:u, 8}) do
      flunk("""
      expected

      #{inspect(lhs)}

      to be within tolerance of

      #{inspect(rhs)}
      """)
    end
  end
end
