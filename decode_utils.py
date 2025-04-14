
import torch
import torch.nn.functional as F
from translation_model import create_subsequent_mask

# Beam Search Decoding (batch size = 1)
def beam_search_decode(model, src, src_vocab, tgt_vocab, beam_width=5, max_len=30):
    device = src.device
    src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
    memory = model.encode(src, src_mask)

    sequences = [[torch.tensor([1], device=device), 0]]  # ([<sos>], log_prob)

    for _ in range(max_len):
        all_candidates = []
        for seq, score in sequences:
            tgt_mask = create_subsequent_mask(len(seq)).unsqueeze(0).to(device)
            out = model.decode(seq.unsqueeze(0), memory, tgt_mask, src_mask)
            probs = F.log_softmax(out[:, -1], dim=-1)

            topk = torch.topk(probs, beam_width)
            for i in range(beam_width):
                next_token = topk.indices[0][i].item()
                next_score = topk.values[0][i].item()
                new_seq = torch.cat([seq, torch.tensor([next_token], device=device)])
                all_candidates.append((new_seq, score + next_score))

        sequences = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)[:beam_width]
        if all(seq[-1] == 2 for seq, _ in sequences):
            break

    best_seq = sequences[0][0].tolist()
    return tgt_vocab.decode(best_seq)


def greedy_decode(model, src, src_vocab, tgt_vocab, max_len=30):
    device = src.device
    src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
    memory = model.encode(src, src_mask)

    ys = torch.ones(1, 1).fill_(1).long().to(device)  # <sos>
    for _ in range(max_len):
        tgt_mask = create_subsequent_mask(ys.size(1)).to(device).unsqueeze(0)
        out = model.decode(ys, memory, tgt_mask, src_mask)
        next_token = out[:, -1].argmax(-1).unsqueeze(1)
        ys = torch.cat([ys, next_token], dim=1)
        if next_token.item() == 2:
            break

    return tgt_vocab.decode(ys[0].tolist())


def translate(model, sentence, src_vocab, tgt_vocab, method="beam", beam_width=5):
    model.eval()
    with torch.no_grad():
        if isinstance(sentence, str):
            src_tensor = torch.tensor([src_vocab.encode(sentence)]).to(next(model.parameters()).device)
        else:
            src_tensor = sentence.to(next(model.parameters()).device)

        if method == "beam":
            return beam_search_decode(model, src_tensor, src_vocab, tgt_vocab, beam_width=beam_width)
        else:
            return greedy_decode(model, src_tensor, src_vocab, tgt_vocab)
