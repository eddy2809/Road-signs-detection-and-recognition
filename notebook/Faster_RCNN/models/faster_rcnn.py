import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_model(num_classes,imgsz):
    """
    Crea un modello Faster R-CNN con backbone ResNet-50-FPN, pre-addestrato sul set di dati COCO,
    personalizzato per un numero specificato di classi e dimensioni dell'immagine.

    Argomenti:
        num_classes (int): il numero di classi per il rilevamento, compreso lo sfondo.
        imgsz (int): la dimensione dell'immagine per il ridimensionamento delle immagini di input.

    Restituisce:
        torchvision.models.detection.FasterRCNN: un modello Faster R-CNN pronto per l'addestramento o l'inferenza.

    """

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True,min_size=imgsz,max_size=imgsz)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model