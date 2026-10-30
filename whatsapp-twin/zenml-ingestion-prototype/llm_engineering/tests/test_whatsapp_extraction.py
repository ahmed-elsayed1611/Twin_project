from llm_engineering.application.dataset.whatsapp_extractor import WhatsAppExtractor

def test_time_gap_detection(tmp_path):
    # Setup extractor with 1 hour max gap
    extractor = WhatsAppExtractor(self_sender="Ahmed", max_time_gap_hours=1)
    
    # Create a chat file with a 5-hour gap
    f = tmp_path / "gap_chat.txt"
    f.write_text(
        "22/03/2023, 10:00 AM - Contact: Good morning\n"
        "22/03/2023, 03:00 PM - Ahmed: Good afternoon", # 5 hours later
        encoding="utf-8"
    )
    
    samples = extractor.extract_samples(str(f))
    # Should be 0 because the gap is too large to consider "Good morning" as context for "Good afternoon"
    assert len(samples) == 0

def test_persona_injection(tmp_path):
    persona = "You are a helpful twin."
    extractor = WhatsAppExtractor(self_sender="Ahmed", persona_context=persona)
    
    f = tmp_path / "persona_chat.txt"
    f.write_text(
        "22/03/2023, 10:00 AM - Contact: Hello\n"
        "22/03/2023, 10:01 AM - Ahmed: Hi there",
        encoding="utf-8"
    )
    
    samples = extractor.extract_samples(str(f))
    assert len(samples) == 1
    assert persona in samples[0].instruction
    assert "Ahmed:" in samples[0].instruction

def test_arabic_digit_normalization():
    extractor = WhatsAppExtractor()
    # Test internal timestamp parser
    dt = extractor._parse_timestamp("٢٢/٠٣/٢٠٢٣", "١٠:٠٠", "ص")
    assert dt.day == 22
    assert dt.month == 3
    assert dt.year == 2023
    assert dt.hour == 10
