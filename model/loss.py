import torch.nn.functional as F
import torch.nn as nn
import torch

def cascade_split_loss(output, target):
    
    output_fuel, output_config, output_cyl, output_turbo, \
    output_misfire, output_loc, output_idle, output_make, \
    output_oem, output_disp_cls, output_hp_cls,  = output
    
    target = torch.split(target, 1, dim=2)

    target_fuel, target_config, target_cyl, target_turbo, \
    target_misfire, target_loc, target_idle, target_make, \
    target_oem, target_disp_cls, target_hp_cls  = target

    loss_fuel = F.nll_loss(output_fuel, torch.flatten(target_fuel), ignore_index=-100)
    loss_config = F.nll_loss(output_config, torch.flatten(target_config), ignore_index=-100)
    loss_cyl = F.nll_loss(output_cyl, torch.flatten(target_cyl), ignore_index=-100)
    loss_turbo = F.nll_loss(output_turbo, torch.flatten(target_turbo), ignore_index=-100)
    loss_misfire = F.nll_loss(output_misfire, torch.flatten(target_misfire), ignore_index=-100)
    
    loss_loc = F.nll_loss(output_loc, torch.flatten(target_loc), ignore_index=-100)
    loss_idle = F.nll_loss(output_idle, torch.flatten(target_idle), ignore_index=-100)
    loss_make = F.nll_loss(output_make, torch.flatten(target_make), ignore_index=-100)
    loss_oem = F.nll_loss(output_oem, torch.flatten(target_oem), ignore_index=-100)
    loss_disp_cls = F.nll_loss(output_disp_cls, torch.flatten(target_disp_cls), ignore_index=-100)
    loss_hp_cls = F.nll_loss(output_hp_cls, torch.flatten(target_hp_cls), ignore_index=-100)

    overall_loss  = loss_fuel + loss_config + loss_cyl + loss_turbo + loss_misfire
    
    loss_dict = {'fuel': loss_fuel, 'config': loss_config, 'cyl': loss_cyl,
                 'turbo': loss_turbo, 'misfire': loss_misfire, 'loc': loss_loc, 
                 'idle': loss_idle, 'make': loss_make, 'oem': loss_oem, 
                 'disp-cls': loss_disp_cls, 'hp-cls': loss_hp_cls,
                  } 

    return overall_loss, loss_dict