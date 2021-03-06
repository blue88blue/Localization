from model.Unet import Unet
from model.AttUnet import AttUnet
from model.PSPNet import PSPNet
from model.DeepLabV3 import DeepLabV3
from model.DANet import DANet
from model.CPFNet import CPFNet
from model.AGNet.model import AG_Net
from model.cenet import CE_Net_
from model.deeplabv3_plus import DeepLabV3Plus




def seg_model(args):
    if args.network == "Unet":
        model = Unet(args.in_channel, args.n_class, channel_reduction=args.Ulikenet_channel_reduction, aux=args.aux)
    elif args.network == "AttUnet":
        model = AttUnet(args.in_channel, args.n_class, channel_reduction=args.Ulikenet_channel_reduction, aux=args.aux)
    elif args.network == "PSPNet":
        model = PSPNet(args.n_class, args.backbone, aux=args.aux, pretrained_base=args.pretrained, dilated=args.dilated, deep_stem=args.deep_stem)
    elif args.network == "DeepLabV3":
        model = DeepLabV3(args.n_class, args.backbone, aux=args.aux, pretrained_base=args.pretrained, dilated=args.dilated, deep_stem=args.deep_stem)
    elif args.network == "DANet":
        model = DANet(args.n_class, args.backbone, aux=args.aux, pretrained_base=args.pretrained, dilated=args.dilated, deep_stem=args.deep_stem)
    elif args.network == "CPFNet":
        model = CPFNet(args.n_class, args.backbone, aux=args.aux, pretrained_base=args.pretrained, dilated=args.dilated, deep_stem=args.deep_stem)
    elif args.network == "AG_Net":
        model = AG_Net(args.n_class)
    elif args.network == "CENet":
        model = CE_Net_(args.n_class)
    elif args.network == "DeepLabV3Plus":
        model = DeepLabV3Plus(args.n_class)
    else:
        NotImplementedError("not implemented {args.network} model")

    return model














