import re
import hashlib
import logging
from datetime import datetime
from typing import Dict, Generator, Iterable, List, Tuple
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class ConversationTurn(BaseModel):
    timestamp: datetime
    sender: str
    content: str
    is_valid: bool = True

class InstructDatasetSample(BaseModel):
    instruction: str
    response: str
    source: str = "whatsapp"
    metadata: Dict = Field(default_factory=dict)

class WhatsAppExtractor:
    """
    Turns a raw WhatsApp export into instruction/response samples for a persona model.

    - Time-gap awareness: context is reset after a long silence (default 4 hours).
    - PII redaction: phone numbers, emails and URLs are replaced with placeholders.
    - Generator-based parsing: the export is streamed, never loaded whole.
    - Persona prompt: a persona instruction is prepended to every sample.
    """

    def __init__(
        self, 
        self_sender: str = "Ahmed",
        anonymize: bool = True,
        min_length: int = 3,
        context_window: int = 5,
        max_time_gap_hours: int = 4,
        persona_context: str = "You are Ahmed, a software engineer and AI enthusiast. Respond in your natural conversational style, mixing Arabic and English as you usually do."
    ):
        self.self_sender = self_sender
        self.anonymize = anonymize
        self.min_length = min_length
        self.context_window = context_window
        self.max_time_gap_hours = max_time_gap_hours
        self.persona_context = persona_context
        
        # Localized WhatsApp header: [date] [time] [AM/PM] - [sender]: [content]
        # Tolerates Arabic-Indic digits, localized separators and invisible direction marks.
        self.header_pattern = re.compile(
            r'^[\u200e\u200f]*.*?([\d\u0660-\u0669]+.*?[\d\u0660-\u0669]+.*?[\d\u0660-\u0669]+).*?([\d\u0660-\u0669]+:[\d\u0660-\u0669]+)\s+([صمAPM]+)\s+-\s+(.*?):\s+(.*)$'
        )
        
        # PII Patterns
        self.phone_pattern = re.compile(r'\+?\d{10,15}')
        self.email_pattern = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
        self.url_pattern = re.compile(r'https?://\S+')
        
        # Noise Markers
        self.noise_markers = [
            "<تم استبعاد الوسائط>", "<Media omitted>",
            "الرسائل والمكالمات مشفرة تمامًا", "Messages and calls are end-to-end encrypted",
            "لقد حذفت أنت هذه الرسالة", "You deleted this message",
            "حذف هذا الشخص هذه الرسالة", "This message was deleted"
        ]

    def _parse_timestamp(self, date_str: str, time_str: str, period: str) -> datetime:
        """Normalizes Arabic/English timestamps into Python datetime objects."""
        # Convert Arabic digits to standard digits
        arabic_digits = "٠١٢٣٤٥٦٧٨٩"
        std_digits = "0123456789"
        trans = str.maketrans(arabic_digits, std_digits)
        
        date_str = date_str.translate(trans).replace('،', ',')
        time_str = time_str.translate(trans)
        
        # Handle Arabic period markers
        period = "AM" if period in ["ص", "AM"] else "PM"
        
        # Try common date formats
        for fmt in ["%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d", "%d.%m.%Y"]:
            try:
                dt_str = f"{date_str} {time_str} {period}"
                return datetime.strptime(dt_str, f"{fmt} %I:%M %p")
            except ValueError:
                continue
        
        # Fallback to current time if parsing fails (rare)
        return datetime.now()

    def _clean_content(self, text: str) -> Tuple[str, bool]:
        """Drops system/media notices, redacts PII and normalizes whitespace."""
        for marker in self.noise_markers:
            if marker in text:
                return "", False
        
        # Redact PII
        text = self.phone_pattern.sub("[PHONE]", text)
        text = self.email_pattern.sub("[EMAIL]", text)
        text = self.url_pattern.sub("[URL]", text)
        
        # Clean edit tags
        text = re.sub(r'\s*<تم تعديلها>$', '', text)
        text = re.sub(r'\s*<Edited>$', '', text)
        
        cleaned = " ".join(text.split()) # Normalize whitespace
        return cleaned, len(cleaned) >= self.min_length

    def iter_turns(self, file_path: str) -> Generator[ConversationTurn, None, None]:
        """Memory-efficient generator for turns."""
        current_turn = None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                
                match = self.header_pattern.match(line)
                if match:
                    if current_turn:
                        content, is_valid = self._clean_content(current_turn.content)
                        if is_valid:
                            current_turn.content = content
                            yield current_turn
                    
                    date_part, time_part, period, sender, content = match.groups()
                    dt = self._parse_timestamp(date_part, time_part, period)
                    
                    if self.anonymize and sender != self.self_sender:
                        sender = "Contact"
                    
                    current_turn = ConversationTurn(
                        timestamp=dt,
                        sender=sender,
                        content=content
                    )
                else:
                    if current_turn:
                        current_turn.content += " " + line
            
            if current_turn:
                content, is_valid = self._clean_content(current_turn.content)
                if is_valid:
                    current_turn.content = content
                    yield current_turn

    def extract_samples(self, file_path: str) -> List[InstructDatasetSample]:
        """Parses an export file and builds multi-turn samples from it."""
        return self.extract_samples_from_turns(self.iter_turns(file_path))

    def extract_samples_from_turns(self, turns: Iterable[ConversationTurn]) -> List[InstructDatasetSample]:
        """Builds one sample for every reply by the self sender that follows a contact message."""
        samples = []
        history: List[ConversationTurn] = []
        seen_hashes = set()

        for turn in turns:
            # A long silence starts a new conversation: earlier messages are not context.
            if history and (turn.timestamp - history[-1].timestamp).total_seconds() > (self.max_time_gap_hours * 3600):
                history = []

            if turn.sender == self.self_sender:
                if history and history[-1].sender != self.self_sender:
                    context_str = "\n".join([f"{t.sender}: {t.content}" for t in history[-self.context_window:]])
                    full_instruction = f"{self.persona_context}\n\nConversation History:\n{context_str}\n\nAhmed:"

                    pair_hash = hashlib.md5(f"{full_instruction}{turn.content}".encode()).hexdigest()
                    if pair_hash not in seen_hashes:
                        samples.append(InstructDatasetSample(
                            instruction=full_instruction,
                            response=turn.content,
                            metadata={
                                "timestamp": turn.timestamp.isoformat(),
                                "context_depth": len(history[-self.context_window:])
                            }
                        ))
                        seen_hashes.add(pair_hash)

            history.append(turn)
            if len(history) > self.context_window:
                history.pop(0)

        return samples
