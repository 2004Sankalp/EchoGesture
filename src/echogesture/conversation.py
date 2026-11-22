"""Human confirmation controls. No language-model rewriting or auto-speech."""
from dataclasses import dataclass, field


@dataclass
class Conversation:
    caption: list = field(default_factory=list)
    pending: dict | None = None
    status: str = 'Ready. Ask the signer before starting.'

    def propose(self, result, visible=True):
        if not visible:
            self.pending = None
            self.status = 'Hands not visible. Please try again.'
            return
        self.pending = result
        # This is a heuristic UI cue, not a calibrated correctness estimate.
        self.status = ('Please check this word.' if result['confidence'] >= 0.65
                       else 'I am unsure. Choose a word or try again.')

    def confirm(self, rank=0):
        if self.pending is None:
            return False
        options = self.pending['top5']
        if not 0 <= rank < len(options):
            return False
        self.caption.append(options[rank])
        self.pending = None
        self.status = 'Added. You can undo the last word.'
        return True

    def undo(self):
        if self.caption:
            self.caption.pop()

    def clear(self):
        self.caption.clear()
        self.pending = None
        self.status = 'Conversation cleared from memory.'
