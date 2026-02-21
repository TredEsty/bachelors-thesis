from django.shortcuts import render
from .services import MovieService

def index(request):
    analysis = None
    results = []
    
    query = request.GET.get('query', '')
    tconst = request.GET.get('tconst')
    
    if query and not tconst:
        results = MovieService.search_movies(query)
        
    elif query and tconst:
        analysis = MovieService.analyze_movie(tconst)
    
    return render(request, 'analyzer/index.html', {
        'analysis': analysis,
        'results': results,
        'query': query
    })