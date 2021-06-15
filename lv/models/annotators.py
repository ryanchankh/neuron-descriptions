"""Models for annotating new, unseen neurons."""
from typing import (Any, Mapping, NamedTuple, Optional, Sequence, Sized, Tuple,
                    Type, TypeVar, cast, overload)

from lv.models import featurizers
from lv.utils import lang, serialize, training
from lv.utils.typing import Device, StrSequence

import numpy
import torch
from sklearn import metrics
from torch import nn, optim
from torch.utils import data
from tqdm.auto import tqdm


class WordClassifierHead(serialize.SerializableModule):
    """Classifier that predicts word distribution from visual features."""

    def __init__(self, feature_size: int, vocab_size: int):
        """Initialize the model.

        Args:
            feature_size (int): Visual feature size..
            vocab_size (int): Vocab size.

        """
        super().__init__()

        self.feature_size = feature_size
        self.vocab_size = vocab_size

        self.classifier = nn.Linear(feature_size, vocab_size)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict the words that describe the images and masks.

        Args:
            features (torch.Tensor): Unit top images. Should have shape
                (batch_size, n_top_images, feature_size).

        Returns:
            torch.Tensor: Shape (batch_size, vocab_size) tensor containing
                probability each word describes the top images.

        """
        logits = self.classifier(features)
        predictions = torch.sigmoid(logits).mean(dim=1)
        return predictions

    def properties(self) -> serialize.Properties:
        """Override `Serializable.properties`."""
        return {
            'feature_size': self.feature_size,
            'vocab_size': self.vocab_size,
        }


class WordAnnotations(NamedTuple):
    """Word annotations predicted by our model."""

    # Probabilities for *every word*, even those not predicted.
    # Will have shape like (batch_size, vocab_size).
    probabilities: torch.Tensor

    # Predicted words and corresponding indices. Each has length
    # of batch_size. Length of internal lists could be anything.
    words: Sequence[StrSequence]
    indices: Sequence[Sequence[int]]


WordAnnotatorT = TypeVar('WordAnnotatorT', bound='WordAnnotator')


class WordAnnotator(serialize.SerializableModule):
    """Predicts words that would appear in the caption of masked images."""

    def __init__(self,
                 indexer: lang.Indexer,
                 featurizer: featurizers.Featurizer,
                 classifier: Optional[WordClassifierHead] = None):
        """Initialize the model.

        Args:
            featurizer (featurizers.Featurizer): Model mapping images and masks
                to visual features.
            indexer (lang.Indexer): Indexer mapping words to indices.
            classifier (Optional[WordClassifierHead], optional): The
                classification head. If not specified, new one is initialized.

        """
        super().__init__()

        if classifier is None:
            feature_size = numpy.prod(featurizer.feature_shape).item()
            vocab_size = len(indexer.vocab)
            classifier = WordClassifierHead(feature_size, vocab_size)

        self.indexer = indexer
        self.featurizer = featurizer
        self.classifier = classifier

    @property
    def feature_size(self) -> int:
        """Return feature size."""
        return self.classifier.classifier.in_features

    @property
    def vocab_size(self) -> int:
        """Return vocab size."""
        return self.classifier.classifier.out_features

    @overload
    def forward(self,
                images: torch.Tensor,
                masks: torch.Tensor,
                threshold: float = ...,
                **kwargs: Any) -> WordAnnotations:
        """Predict the words that describe the images and masks.

        Args:
            images (torch.Tensor): Unit top images. Should have shape
                (batch_size, n_top_images, 3, height, width).
            masks (torch.Tensor): Unit top image masks. Should have shape
                (batch_size, n_top_images, 1, height, width).
            threshold (float, optional): Cutoff for whether or not a word
                is predicted or not.

        Returns:
            WordAnnotations: Predicted annotations for images/masks.

        """
        ...

    @overload
    def forward(self, images: torch.Tensor, **kwargs: Any) -> WordAnnotations:
        """Predict the words that describe the given image features.

        Keyword arguments are the same as in other signature.

        Args:
            images (torch.Tensor): Image features. Should have shape
                (batch_size, *featurizer.feature_shape).

        Returns:
            WordAnnotations: Predicted annotations for features.

        """
        ...

    def forward(self,
                images: torch.Tensor,
                masks: Optional[torch.Tensor] = None,
                threshold: float = .5,
                **_: Any) -> WordAnnotations:
        """Implement overloaded functions above."""
        batch_size = images.shape[0]
        if masks is not None:
            images = images.view(-1, 3, *images.shape[-2:])
            masks = masks.view(-1, 1, *masks.shape[-2:])
            features = self.featurizer(images, masks)
        else:
            features = images
        features = features.view(batch_size, -1, self.feature_size)

        # Compute probability of each word.
        probabilities = self.classifier(features)

        # Dereference words with probabilities over the threshold, sorted
        # by their probability.
        words, indices = [], []
        for s_ps in probabilities:
            s_indices = s_ps.gt(threshold)\
                .nonzero(as_tuple=False)\
                .squeeze()\
                .tolist()

            # Sometimes this chain of calls returns a single element. Rewrap it
            # in a list for consistency.
            if isinstance(s_indices, int):
                s_indices = [s_indices]

            s_indices = sorted(s_indices,
                               key=lambda index: s_ps[index].item(),
                               reverse=True)
            s_words = self.indexer.unindex(s_indices)
            words.append(s_words)
            indices.append(s_indices)

        return WordAnnotations(probabilities, tuple(words), tuple(indices))

    def f1(self,
           dataset: data.Dataset,
           annotation_index: int = 4,
           threshold: float = .5,
           predictions: Optional[WordAnnotations] = None,
           **kwargs: Any) -> Tuple[float, WordAnnotations]:
        """Compute F1 score of this model on the given dataset.

        Keyword arguments are forwarded to `forward`.

        Args:
            dataset (data.Dataset): Test dataset.
            annotation_index (int, optional): Index of annotations in dataset
                samples. Defaults to 4 to be compatible with
                AnnotatedTopImagesDataset.
            threshold (float, optional): Probability threshold for whether or
                not the model predicts a word. Defaults to .5.
            predictions (Optional[WordAnnotations], optional): Precomputed
                word predictions. Defaults to None.

        Returns:
            float: F1 score.

        """
        if predictions is None:
            predictions = self.predict(dataset, threshold=threshold, **kwargs)
        y_pred = predictions.probabilities.gt(threshold).int().cpu().numpy()

        annotations = []
        for index in range(len(y_pred)):
            annotation = dataset[index][annotation_index]
            annotation = lang.join(annotation)
            annotations.append(annotation)

        y_true = numpy.zeros((len(y_pred), len(self.indexer.vocab)))
        for index, annotation in enumerate(annotations):
            indices = self.indexer(annotation)
            y_true[index, sorted(set(indices))] = 1

        f1 = metrics.f1_score(y_pred=y_pred,
                              y_true=y_true,
                              average='weighted',
                              zero_division=0.)
        return f1

    def predict(self,
                dataset: data.Dataset,
                mask: bool = True,
                image_index: int = 2,
                mask_index: int = 3,
                batch_size: int = 16,
                features: Optional[data.TensorDataset] = None,
                num_workers: int = 0,
                device: Optional[Device] = None,
                display_progress_as: Optional[str] = 'predict words',
                **kwargs: Any) -> WordAnnotations:
        """Feed entire dataset through the annotation model.

        Keyword arguments are passed to forward.

        Args:
            dataset (data.Dataset): The dataset of images/masks.
            mask (bool, optional): Use masks when computing features. Exact
                behavior depends on the featurizer. Defaults to True.
            image_index (int, optional): Index of images in dataset samples.
                Defaults to 2 to be compatible with AnnotatedTopImagesDataset.
            mask_index (int, optional): Index of masks in dataset samples.
                Defaults to 3 to be compatible with AnnotatedTopImagesDataset.
            batch_size (int, optional): Number of samples to process on at
                once. Defaults to 16.
            features (Optional[data.TensorDataset], optional): Precomputed
                image features. Defaults to None.
            num_workers (int, optional): Number of workers for loading data.
                Defaults to 0.
            device (Optional[Device], optional): Send model and data to this
                device. Defaults to None.
            display_progress_as (Optional[str], optional): Show a progress bar
                with this key when predicting words. Defaults to
                'predict words'.

        Returns:
            WordAnnotations: Predicted annotations for every sample in dataset.

        """
        if device is not None:
            self.to(device)
        if features is None:
            features = self.featurizer.map(
                dataset,
                mask=mask,
                image_index=image_index,
                mask_index=mask_index,
                batch_size=batch_size,
                device=device,
                display_progress_as=display_progress_as is not None)

        loader = data.DataLoader(features,
                                 batch_size=batch_size,
                                 num_workers=num_workers)
        if display_progress_as is not None:
            loader = tqdm(loader, desc=display_progress_as)

        predictions = []
        for (inputs,) in loader:
            with torch.no_grad():
                outputs = self(inputs, **kwargs)
            predictions.append(outputs)

        probabilities = torch.cat([pred.probabilities for pred in predictions])

        words, indices = [], []
        for prediction in predictions:
            words.extend(prediction.words)
            indices.extend(prediction.indices)

        return WordAnnotations(probabilities, tuple(words), tuple(indices))

    def fit(
        self,
        dataset: data.Dataset,
        mask: bool = True,
        image_index: int = 2,
        mask_index: int = 3,
        annotation_index: int = 4,
        batch_size: int = 64,
        max_epochs: int = 1000,
        patience: Optional[int] = None,
        optimizer_t: Type[optim.Optimizer] = optim.Adam,
        optimizer_kwargs: Optional[Mapping[str, Any]] = None,
        features: Optional[data.TensorDataset] = None,
        num_workers: int = 0,
        device: Optional[Device] = None,
        display_progress_as: Optional[str] = 'train word annotator',
    ) -> None:
        """Train a new WordAnnotator from scratch.

        Args:
            dataset (data.Dataset): Training dataset.
            mask (bool, optional): Use masks when computing features. Exact
                behavior depends on the featurizer. Defaults to True.
            image_index (int, optional): Index of images in dataset samples.
                Defaults to 2 to be compatible with AnnotatedTopImagesDataset.
            mask_index (int, optional): Index of masks in dataset samples.
                Defaults to 3 to be compatible with AnnotatedTopImagesDataset.
            annotation_index (int, optional): Index of language annotations in
                dataset samples. Defaults to 4 to be compatible with
                AnnotatedTopImagesDataset.
            batch_size (int, optional): Number of samples to train on at once.
                Defaults to 64.
            max_epochs (int, optional): Maximum number of epochs to train for.
                Defaults to 1000.
            patience (Optional[int], optional): If validation loss does not
                improve for this many epochs, stop training. By default, no
                early stopping.
            optimizer_t (Type[optim.Optimizer], optional): Optimizer to use.
                Defaults to Adam.
            optimizer_kwargs (Optional[Mapping[str, Any]], optional): Optimizer
                keyword arguments to pass at construction. Defaults to None.
            features (Optional[data.TensorDataset], optional): Precomputed
                image features. By default, computed from the full dataset.
            num_workers (int, optional): Number of workers for loading data.
                Defaults to 0.
            device (Optional[Device], optional): Send model and all data to
                this device. Defaults to None.
            display_progress_as (Optional[str], optional): Show a progress bar
                with this key when training model.
                Defaults to 'train word annotator'.

        """
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        if features is None:
            features = self.featurizer.map(
                dataset,
                mask=mask,
                image_index=image_index,
                mask_index=mask_index,
                batch_size=batch_size,
                device=device,
                display_progress_as=display_progress_as is not None)

        targets = torch.zeros(len(features),
                              len(self.indexer.vocab),
                              device=device)
        for index in range(len(features)):
            annotation = dataset[index][annotation_index]
            annotation = lang.join(annotation)
            indices = self.indexer(annotation)
            targets[index, sorted(set(indices))] = 1

        features_loader = data.DataLoader(features,
                                          num_workers=num_workers,
                                          batch_size=batch_size)
        targets_loader = data.DataLoader(data.TensorDataset(targets),
                                         num_workers=num_workers,
                                         batch_size=batch_size)

        classifier = self.classifier.classifier

        optimizer_kwargs = dict(optimizer_kwargs)
        optimizer_kwargs.setdefault('lr', 1e-4)
        optimizer = optimizer_t(classifier.parameters(), **optimizer_kwargs)

        stopper = None
        if patience is not None:
            stopper = training.EarlyStopping(patience=patience)

        # Balance the dataset using the power of BAYESIAN STATISTICS, BABY!
        n_positives = targets.sum(dim=0)
        n_negatives = len(targets) - n_positives
        pos_weight = n_negatives / n_positives
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        progress = range(max_epochs)
        if display_progress_as is not None:
            progress = tqdm(progress, desc=display_progress_as)

        for _ in progress:
            train_loss = 0.
            for (inputs,), (targets,) in zip(features_loader, targets_loader):
                inputs = inputs.view(*inputs.shape[:2], -1)
                targets = targets[:, None, :].expand(-1, inputs.shape[1], -1)
                predictions = classifier(inputs)
                loss = criterion(predictions, targets)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                train_loss += loss.item()
            train_loss /= len(features_loader)

            if display_progress_as is not None:
                assert not isinstance(progress, range)
                progress.set_description(f'{display_progress_as} '
                                         f'[loss={train_loss:.3f}]')

            if stopper is not None and stopper(train_loss):
                break

    def properties(self) -> serialize.Properties:
        """Override `Serializable.properties`."""
        return {
            'indexer': self.indexer,
            'featurizer': self.featurizer,
            'classifier': self.classifier,
        }

    def serializable(self) -> serialize.Children:
        """Override `Serializable.serializable`."""
        return {'featurizer': featurizers.key(self.featurizer)}

    @classmethod
    def resolve(cls, children: serialize.Children) -> serialize.Resolved:
        """Override `Serializable.resolve`."""
        key = children.get('featurizer')
        assert key is not None
        return {'featurizer': featurizers.parse(key)}


def word_annotator(dataset: data.Dataset,
                   featurizer: featurizers.Featurizer,
                   annotation_index: int = 4,
                   indexer_kwargs: Optional[Mapping[str, Any]] = None,
                   **kwargs: Any) -> WordAnnotator:
    """Create a new word annotator from the given dataset and image featurizer.

    Keyword arguments are forwarded to the constructor.

    Args:
        dataset (data.Dataset): Training dataset.
        featurizer (featurizers.Featurizer): Image featurizer.
        annotation_index (int, optional): Index of language annotations in
            dataset samples. Defaults to 4 to be compatible with
            AnnotatedTopImagesDataset.
        indexer_kwargs (Optional[Mapping[str, Any]], optional): Indexer
            keyword arguments to pass at construction. Defaults to None.

    Returns:
        WordAnnotator: The instantiated `WordAnnotator`.

    """
    if indexer_kwargs is None:
        indexer_kwargs = {}

    annotations = []
    for index in range(len(cast(Sized, dataset))):
        annotation = dataset[index][annotation_index]
        annotation = lang.join(annotation)
        annotations.append(annotation)

    indexer_kwargs = dict(indexer_kwargs)
    indexer_kwargs.setdefault('ignore_rarer_than', 1)
    indexer = lang.indexer(annotations, **indexer_kwargs)

    return WordAnnotator(indexer, featurizer, **kwargs)
