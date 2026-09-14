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
