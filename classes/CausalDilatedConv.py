from torch import nn

class DilatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(DilatedConv, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation, padding=self.padding, bias=False)
        
    def forward(self, x):
        out = self.conv(x)
        
        return out[:, :, :-self.padding]
    
        #  Exemple pour dilation = 1
        # 
        #  | 0 | a | b | c | d | e | f | g | h | 0 |
        #    |  /|  /|  /|  /|  /|  /|  /|  /|  /
        #    | / | / | / | / | / | / | / | / | /
        #  | 0a| ab| bc| cd| de| ef| fg| gh| h0|
        # 
        # Les valeurs dans chaque case de sortie correspondent à ce que "voit" une cellule après la convolution
        # Si on veut une convolution causale, alors chaque case d'arrivée (exemple "0a") correspond à la convolution 
        # pour la dernière valeur de cette cellule, donc "a". La séquence est paddée de (kernel - 1) pour exactement 
        # pour que le premier timestep de la sortie corresponde à la convolution causale du premier timestep de l'entrée.
        # En conséquence, les (kernel - 1) derniers timestep de la sortie correspondent aux convolutions causales pour les
        # cellules ajoutées par padding à la fin de la séquence. Elles ne servent donc à rien, c'est pourquoi on coupe les
        # (kernel - 1) derniers timesteps de la séquence de sortie, cela conserve la longueur de la séquence.
        # 
        #  | 0a| ab| bc| cd| de| ef| fg| gh| h0|
        #    |   |   |   |   |   |   |   |
        #  | 0a| ab| bc| cd| de| ef| fg| gh|
        # 
        # Lorsque la dilation est plus grande que 1 on prend un padding égal à (kernel - 1) * dilation.