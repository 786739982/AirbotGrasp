from segment_anything import sam_model_registry, SamPredictor

class AirbotSegment():

    def __init__(self) -> None:
        self.sam_checkpoint = './checkpoint/sam_vit_b.pth'
        self.model_type = 'vit_b'
        self.device = 'cuda'
        self.sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        self.sam.to(self.device)
        self.predictor = SamPredictor(self.sam)
    
    def get_model(self):
        return self.predictor