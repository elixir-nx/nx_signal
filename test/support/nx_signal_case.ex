defmodule NxSignal.Case do
  use ExUnit.CaseTemplate

  using do
    quote do
      import NxSignal.Case

      test "defines doc :type" do
        validate_doc_metadata(__MODULE__)
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
end
