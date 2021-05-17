import collections
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from pytorch_pretrained_bert.tokenization import convert_to_unicode, BertTokenizer


class InputExample(object):

    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def read_examples(list_of_rows):
    """
    Takes in a list of rows with row format either [line1,line2] or just [line1]
    """
    examples = []
    unique_id = 0
    for row in list_of_rows:
        # Simple conversion to unicode for processing
        for i in range(len(row)):
            row[i] = convert_to_unicode(row[i])

        text_a = None
        text_b = None
        if len(row) == 1:
            text_a = row[0]
        elif len(row) > 2 or len(row) < 1:
            raise ValueError("Atleast 1 element is required inside the row and not more than two elements are valid")
        elif len(row) == 2:
            text_a = row[0]
            text_b = row[1]
        examples.append(
            InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
        unique_id += 1
    return examples


def get_features(model, model_type, list_of_rows, max_seq_length, device, batch_size, layers="-1,-2,-3,-4"):
    """
    :param model: bert model
    :param model_type: "Bert pre-trained model selected in the list: bert-base-uncased, bert-large-uncased, bert-base-cased,
                        bert-base-multilingual, bert-base-chinese."
    :param list_of_rows: list of rows for which the features has to be calculated
    :param max_seq_length: max seq length of the model
    :param device: cpu or gpu
    :param batch_size: batch size
    :param layers: layers for which the features has to be calculated, -1 indicates last layer, -2 indicates second last and so on
    :return:
    """

    all_example_out_features = {}

    layer_indexes = [int(x) for x in layers.split(",")]
    tokenizer = BertTokenizer.from_pretrained(model_type)
    examples = read_examples(list_of_rows)
    features = convert_examples_to_features(
        examples=examples, seq_length=max_seq_length, tokenizer=tokenizer)
    unique_id_to_feature = {}
    for feature in features:
        unique_id_to_feature[feature.unique_id] = feature

    model.to(device)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)

    # gives index to each row of input ids, this will work same as unique ids in features
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

    """
    Now we have all_input_ids = [[input_id_00, input_id_01, input_id_02],
                                 [input_id_10, input_id_11, input_id_12],
                                 [], ...]
    Their masks in all_input_masks, and
    for each row a unique id in all_example_index = [0,1,2,3....]
    """

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_example_index)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)

    model.eval()

    for input_ids, input_mask, example_indices in eval_dataloader:

        # we are getting a batch here
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)

        # send the batch to the model
        all_encoder_layers, pooled_features = model(input_ids, token_type_ids=None, attention_mask=input_mask)

        # going through each example_index in this batch

        for b, example_index in enumerate(example_indices):

            feature = features[example_index.item()]

            unique_id = int(feature.unique_id)
            output_json = collections.OrderedDict()
            output_json["linex_index"] = unique_id
            output_json["pooled_feature"] = pooled_features[b]

            all_out_features = []
            for (i, token) in enumerate(feature.tokens):
                all_layers = []
                for (j, layer_index) in enumerate(layer_indexes):
                    layer_output = all_encoder_layers[int(layer_index)].detach().cpu().numpy()

                    # b in next line is important, that is how its taking feature for particular item of batch
                    layer_output = layer_output[b]
                    layers = collections.OrderedDict()
                    layers["index"] = layer_index
                    layers["values"] = [
                        round(x.item(), 6) for x in layer_output[i]
                    ]
                    all_layers.append(layers)
                out_features = collections.OrderedDict()
                out_features["token"] = token
                out_features["layers"] = all_layers
                all_out_features.append(out_features)

            output_json["features"] = all_out_features
            all_example_out_features[unique_id] = output_json

    return all_example_out_features





