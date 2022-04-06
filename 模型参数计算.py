from torchstat import stat
import torchvision.models as models
from shufflenet import ShuffleNetv2_Att
model = ShuffleNetv2_Att()
stat(model, (3, 256, 256))
print(model)