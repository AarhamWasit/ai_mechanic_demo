import torch
import torch.nn as nn
import torch.nn.functional as F
        
class Cascade_2D_M5(nn.Module):
    def __init__(self, n_input=1, n_output=2, n_channel=32):
        super().__init__()
        self.conv1 = nn.Conv2d(n_input, n_channel, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(n_channel)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.conv1_drop = nn.Dropout2d()

        self.conv2 = nn.Conv2d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(n_channel)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.conv2_drop = nn.Dropout2d()

        self.conv3 = nn.Conv2d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(2 * n_channel)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.conv3_drop = nn.Dropout2d()

        self.fc_fuel = nn.Linear(2 * n_channel, 2)
        self.fc_config = nn.Linear(2 * n_channel, 3)
        self.fc_cyl = nn.Linear(2 * n_channel, 7)
        self.fc_turbo = nn.Linear(2 * n_channel, 2)
        self.fc_loc = nn.Linear(2 * n_channel, 5)
        self.fc_idle = nn.Linear(2 * n_channel, 2)
        self.fc_make = nn.Linear(2 * n_channel, 30)
        self.fc_oem = nn.Linear(2 * n_channel, 19)
        self.fc_disp_cls = nn.Linear(2 * n_channel, 37)
        self.fc_hp_cls = nn.Linear(2 * n_channel, 93)

        self.conv1_m = nn.Conv2d(n_input, n_channel, kernel_size=3)
        self.bn1_m = nn.BatchNorm2d(n_channel)
        self.pool1_m = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.conv1_drop_m = nn.Dropout2d()
        
        self.conv2_m = nn.Conv2d(n_channel, n_channel, kernel_size=3)
        self.bn2_m = nn.BatchNorm2d(n_channel)
        self.pool2_m = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.conv2_drop_m = nn.Dropout2d()
        
        self.conv3_m = nn.Conv2d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3_m = nn.BatchNorm2d(2 * n_channel)
        self.pool3_m = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.conv3_drop_m = nn.Dropout2d()

        self.fc_misfire = nn.Linear(2 * n_channel, 2)

    def forward(self, x_input):
        x_input = x_input.float()
        x = self.conv1(x_input)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv1_drop(x)
        
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv2_drop(x)
        
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv3_drop(x)
        
        #x = x.squeeze()
        x = x.squeeze(-1)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)

        x_fuel = self.fc_fuel(x)
        x_config = self.fc_config(x)
        x_cyl = self.fc_cyl(x)
        x_turbo = self.fc_turbo(x)
        x_loc = self.fc_loc(x)
        x_idle = self.fc_idle(x)
        x_make = self.fc_make(x)
        x_oem = self.fc_oem(x)
        x_disp_cls = self.fc_disp_cls(x)
        x_hp_cls = self.fc_hp_cls(x)

        x_fuel = x_fuel.view(-1, 2)
        x_config = x_config.view(-1, 3)
        x_cyl = x_cyl.view(-1, 7)
        x_turbo = x_turbo.view(-1, 2)
        x_loc = x_loc.view(-1, 5)
        x_idle = x_idle.view(-1, 2)
        x_make = x_make.view(-1, 30)
        x_oem = x_oem.view(-1, 19)
        x_disp_cls = x_disp_cls.view(-1, 37)
        x_hp_cls = x_hp_cls.view(-1, 93)

        x_fuel = F.log_softmax(x_fuel, dim=1)
        x_config = F.log_softmax(x_config, dim=1)
        x_cyl = F.log_softmax(x_cyl, dim=1)
        x_turbo = F.log_softmax(x_turbo, dim=1)
        x_loc = F.log_softmax(x_loc, dim=1)
        x_idle = F.log_softmax(x_idle, dim=1)
        x_make = F.log_softmax(x_make, dim=1)
        x_oem = F.log_softmax(x_oem, dim=1)
        x_disp_cls = F.log_softmax(x_disp_cls, dim=1)
        x_hp_cls = F.log_softmax(x_hp_cls, dim=1)

        out_fuel = torch.argmax(x_fuel, dim=1).unsqueeze(1)
        out_config = torch.argmax(x_config, dim=1).unsqueeze(1)
        out_cyl = torch.argmax(x_cyl, dim=1).unsqueeze(1)
        out_turbo = torch.argmax(x_turbo, dim=1).unsqueeze(1)
        out_loc = torch.argmax(x_loc, dim=1).unsqueeze(1)
        out_idle = torch.argmax(x_idle, dim=1).unsqueeze(1)
        out_make = torch.argmax(x_make, dim=1).unsqueeze(1)
        out_oem = torch.argmax(x_oem, dim=1).unsqueeze(1)
        out_disp_cls = torch.argmax(x_disp_cls, dim=1).unsqueeze(1)
        out_hp_cls = torch.argmax(x_hp_cls, dim=1).unsqueeze(1)

        out_attr = torch.cat((out_fuel, out_config, out_cyl, 
                              out_turbo, out_loc, out_idle,
                              out_make, out_oem, out_disp_cls,
                              out_hp_cls), dim=1)
        
        out_attr = F.pad(out_attr, (0,3))

        out_attr = out_attr.unsqueeze(1).unsqueeze(1)

        misfire_input = torch.cat((x_input, out_attr.float()), dim=2)

        x = self.conv1_m(misfire_input)
        x = F.relu(self.bn1_m(x))
        x = self.pool1_m(x)
        x = self.conv1_drop_m(x)
        
        x = self.conv2_m(x)
        x = F.relu(self.bn2_m(x))
        x = self.pool2_m(x)
        x = self.conv2_drop_m(x)

        x = self.conv3_m(x)
        x = F.relu(self.bn3_m(x))
        x = self.pool3_m(x)
        x = self.conv3_drop_m(x)

        #x = x.squeeze()
        x = x.squeeze(-1)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)

        x_misfire = self.fc_misfire(x)
        x_misfire = x_misfire.view(-1, 2)
        x_misfire = F.log_softmax(x_misfire, dim=1)

        return x_fuel, x_config, x_cyl, x_turbo, x_misfire, \
               x_loc, x_idle, x_make, x_oem, x_disp_cls, x_hp_cls

