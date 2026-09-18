import pytest


def test_extract_sections_assigns_lines_by_pattern(make_index):
    idx = make_index()
    idx.section_schemas["decisao"] = {
        "header": ["poder judiciario"],
        "conclusion": ["decido"],
    }
    text = "\n".join([
        "Texto antes do cabecalho",
        "PODER JUDICIARIO DO ESTADO",
        "Conteudo do cabecalho",
        "Relato dos fatos aqui",
        "Decido.",
        "Texto final da decisao",
    ])

    result = idx.extract_sections(text, "decisao")

    assert result["cabecalho"] == "Texto antes do cabecalho"
    assert result["header"] == (
        "PODER JUDICIARIO DO ESTADO\nConteudo do cabecalho\nRelato dos fatos aqui"
    )
    assert result["conclusion"] == "Decido.\nTexto final da decisao"
    assert result["completo"] == text


def test_extract_sections_matches_patterns_as_whole_words_not_substrings(make_index):
    # Regression test: patterns matched as plain substrings anywhere in a line. On a real
    # corpus, the LLM-calibrated pattern "lar" (for a "header" section) fired inside
    # "declarar" and "solicita" (for "requests") inside "solicitação" — 306 of 514 section
    # switches landed mid-paragraph, scattering prose across unrelated sections.
    idx = make_index()
    idx.section_schemas["edital"] = {
        "header": ["lar", "processo:"],
        "requests": ["solicita"],
        "conclusion": ["decido."],
    }
    text = "\n".join([
        "Preambulo",
        "O candidato devera declarar ciencia das regras.",   # "lar" inside a word
        "Sera admitida a solicitação de inscrição.",          # "solicita" inside a word
        "Processo: 0829366-83.2025",                         # pattern ending in punctuation
        "A parte solicita a gratuidade.",                    # whole word
        "Decido.",
    ])

    result = idx.extract_sections(text, "edital")

    assert result["cabecalho"] == (
        "Preambulo\nO candidato devera declarar ciencia das regras.\nSera admitida a solicitação de inscrição."
    )
    assert result["header"] == "Processo: 0829366-83.2025"
    assert result["requests"] == "A parte solicita a gratuidade."
    assert result["conclusion"] == "Decido."


def test_extract_sections_without_calibrated_schema_raises(make_index):
    idx = make_index()
    with pytest.raises(ValueError):
        idx.extract_sections("qualquer texto", "tipo_nao_calibrado")


def test_register_document_type_parses_llm_schema(make_index, fake_chat_provider):
    idx = make_index()
    fake_chat_provider.responses.append({
        "sections": [
            {"name": "Header Info", "patterns": [" ABC ", "xyz"]},
            {"name": "conclusion", "patterns": ["decido"]},
        ]
    })

    schema = idx.register_document_type("contrato", sample_texts=["texto de exemplo"])

    assert schema == {
        "header_info": ["abc", "xyz"],
        "conclusion": ["decido"],
    }
    assert idx.section_schemas["contrato"] == schema


def test_register_document_type_empty_llm_response_yields_empty_schema(make_index, fake_chat_provider):
    idx = make_index()
    fake_chat_provider.responses.append({"sections": []})

    schema = idx.register_document_type("misto", sample_texts=["amostra heterogenea"])

    assert schema == {}
    assert idx.section_schemas["misto"] == {}


def test_register_document_type_uses_cache_without_recalibrating(make_index, fake_chat_provider):
    idx = make_index()
    idx.section_schemas["contrato"] = {"header": ["cabecalho"]}

    schema = idx.register_document_type("contrato", sample_texts=["nao deveria ser usado"])

    assert schema == {"header": ["cabecalho"]}
    assert fake_chat_provider.responses == []  # no chat call was made


def test_register_document_type_force_recalibrate_ignores_cache(make_index, fake_chat_provider):
    idx = make_index()
    idx.section_schemas["contrato"] = {"header": ["antigo"]}
    fake_chat_provider.responses.append({"sections": [{"name": "novo", "patterns": ["padrao"]}]})

    schema = idx.register_document_type(
        "contrato", sample_texts=["amostra"], force_recalibrate=True
    )

    assert schema == {"novo": ["padrao"]}


# --- a calibration call that failed -----------------------------------------

class _FailingChatProvider:
    """Implements providers.ChatProvider, but the call never gets through."""

    def __init__(self, error=None):
        self.calls = 0
        self.error = error or ConnectionError("connection refused")

    def complete_structured(self, prompt, json_schema):
        self.calls += 1
        raise self.error


def test_register_document_type_raises_and_caches_nothing_when_the_call_fails(make_index):
    # Regression test: the failure used to become an empty schema, which was cached and
    # read as "these documents have no sections" — for the rest of the instance's life,
    # since a cached entry also stops the schema on disk from ever being loaded again.
    provider = _FailingChatProvider()
    idx = make_index(chat_provider=provider)

    with pytest.raises(Exception) as excinfo:
        idx.register_document_type("casos", sample_texts=["amostra"])

    assert "connection refused" in str(excinfo.value)
    assert "casos" not in idx.section_schemas

    # A later call tries again instead of returning the cached failure.
    with pytest.raises(Exception):
        idx.register_document_type("casos", sample_texts=["amostra"])
    assert provider.calls == 2


def test_build_indices_keeps_the_previous_sections_index_when_calibration_fails(
    make_index, fake_chat_provider, fake_embedding_provider, tmp_path, caplog
):
    # The expensive half of the regression: sections wasn't produced, so build_indices
    # deleted the files of the sections index built earlier — a transient outage
    # destroyed vectors that cost one embedding call per section window to rebuild.
    data_dir = tmp_path / "data" / "casos"
    data_dir.mkdir(parents=True)
    (data_dir / "caso1.txt").write_text("Cabecalho\nTrata-se de acao sobre contratos.\n", encoding="utf-8")
    out = tmp_path / "out"

    idx = make_index()
    fake_chat_provider.responses.append({"sections": [{"name": "facts", "patterns": ["trata-se de"]}]})
    idx.build_indices(document_type="casos", base_data_dir=str(tmp_path / "data"), output_index_dir=str(out))
    sections_files = sorted(p.name for p in (out / "casos").glob("casos_sections*"))
    assert sections_files, "the first build should have produced a sections index"

    # Second build, in a fresh process/instance, with the calibration backend down.
    (data_dir / "caso2.txt").write_text("Cabecalho\nTrata-se de acao sobre pagamentos.\n", encoding="utf-8")
    broken = make_index(chat_provider=_FailingChatProvider())
    broken.section_schemas.clear()
    (out / "casos" / "casos_section_schema.json").unlink()  # nothing on disk to fall back to

    with caplog.at_level("ERROR"):
        broken.build_indices(document_type="casos", base_data_dir=str(tmp_path / "data"), output_index_dir=str(out))

    assert "connection refused" in caplog.text
    # full/chunks were rebuilt with both documents...
    assert len(broken.indices["casos"]["full"][1]) == 2
    assert "sections" not in broken.indices["casos"]
    # ...and the earlier sections index is still on disk, rather than deleted.
    assert sorted(p.name for p in (out / "casos").glob("casos_sections*")) == sections_files


def test_build_indices_still_removes_sections_when_the_llm_finds_none(make_index, fake_chat_provider, tmp_path):
    # The other side of the same coin: an empty schema IS an answer about the documents,
    # so a sections index built from a previous corpus must not be left behind.
    data_dir = tmp_path / "data" / "casos"
    data_dir.mkdir(parents=True)
    (data_dir / "caso1.txt").write_text("Cabecalho\nTrata-se de acao sobre contratos.\n", encoding="utf-8")
    out = tmp_path / "out"

    idx = make_index()
    fake_chat_provider.responses.append({"sections": [{"name": "facts", "patterns": ["trata-se de"]}]})
    idx.build_indices(document_type="casos", base_data_dir=str(tmp_path / "data"), output_index_dir=str(out))
    assert list((out / "casos").glob("casos_sections*"))

    rebuilder = make_index()
    fake_chat_provider.responses.append({"sections": []})
    (out / "casos" / "casos_section_schema.json").unlink()
    rebuilder.build_indices(document_type="casos", base_data_dir=str(tmp_path / "data"), output_index_dir=str(out))

    assert not list((out / "casos").glob("casos_sections*"))


def test_calibration_prompt_constrains_the_schema_it_asks_for(make_index, fake_chat_provider):
    # The schema decides how every document of the type is cut, so the prompt has to pin
    # down what comes back: names that are names (a model asked loosely answered with
    # "header_/_identifying_information"), and few enough sections that a document's
    # content isn't spread across a dozen thin vectors.
    from faiss_index import constants

    idx = make_index()
    fake_chat_provider.responses.append({"sections": [{"name": "header", "patterns": ["processo:"]}]})

    idx.register_document_type("casos", sample_texts=["amostra de documento"])

    prompt = fake_chat_provider.prompts[0]
    assert f"at most {constants.MAX_SECTIONS_PER_SCHEMA} sections" in prompt
    assert "snake_case" in prompt
