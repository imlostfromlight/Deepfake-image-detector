from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from .services import DeepfakeModel

class ImageDetectionView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        file = request.data.get('image')
        if not file:
            return Response({"error": "No image uploaded"}, status=400)

        # Initialize and run model
        detector = DeepfakeModel()
        result = detector.predict(file)
        
        return Response(result)