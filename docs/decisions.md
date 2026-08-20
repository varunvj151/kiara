# Kiara Engineering Decisions

Every important design decision should be recorded so that the reasoning behind Kiara remains transparent over time.

## Decision 001

### Tokenizer

Chosen: BPE

Reason:
- Efficient
- Widely used
- Good compression
- Fast

Alternatives:
- SentencePiece
- WordPiece
- Character

Status: Accepted

## Decision 002

### Model Family

Chosen: Decoder-only Transformer

Reason:
- Strong language modeling baseline
- Scales well
- Fits modern foundation model research

Alternatives:
- Encoder-only
- Encoder-decoder

Status: Accepted
