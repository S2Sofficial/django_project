#Self Created file - Swaroop2sky

from django.http import HttpResponse

def index(request):
    return HttpResponse("<h1>This is heading 1</h1> <h2>This is heading 2</h2> <h3>This is heading 3</h3>")

def about(request):
    return HttpResponse("About Page")

