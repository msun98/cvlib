# python 기준.. # c++에서는 Layer::blob을 이용하는거 같던데 # 실제 코드로 확인하지는 않았다.
import cv2 

# load caffemodel 
prototxt = resource_filename(Requirement.parse('cvlib'),
	'cvlib' + os.path.sep + 'data' + os.path.sep + 'deploy.prototxt')
caffemodel = resource_filename(Requirement.parse('cvlib'),
	'cvlib' + os.path.sep + 'data' + os.path.sep + 'res10_300x300_ssd_iter_140000.caffemodel')
            
        # read pre-trained wieights
net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
# get layer name 
layer_names = net.getLayerNames() 

# get params 
layer_params = net.getParam(layerName)