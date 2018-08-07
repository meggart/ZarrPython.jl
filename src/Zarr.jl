__precompile__()
module Zarr
export zopen
using PyCall


function __init__()
  @pyimport zarr
  global zarr = zarr
  global pysli = pybuiltin("slice")
  global pyisi = pybuiltin("isinstance")
  global pylis = pybuiltin("list")
  py"""
  def pycollect(x):
      return [ix for ix in x]
  """
  global pycollect=py"pycollect"

  global ztype2jltype=Dict("<f4"=>Float32, "<f8"=>Float64, "<i4"=>Int32, "<i8"=>Int64)
end

struct ZarrGroup
  g::PyObject
  groups::Vector{String}
  arrays::Vector{String}
end
function Base.show(io::IO, g::ZarrGroup)
  print(io, "Zarr Group with Variables: ", g.arrays, "and groups: ", g.groups)
end
Base.getindex(g::ZarrGroup,s::Union{String,Symbol})=zarrwrap(g.g[Symbol(s)])

struct ZarrArray{T,N}
  v::PyObject
  s::NTuple{N,Int}
  atts::Dict{String}
end
function zarrwrap(o::PyObject)
  if pyisi(o,zarr.hierarchy[:Group])
    arrays = pylis(o[:array_keys]())
    groups = pylis(o[:group_keys]())
    ZarrGroup(o,groups,arrays)
  elseif pyisi(o,zarr.core[:Array])
    s = o[:shape]
    t = ztype2jltype[o[:dtype][:str]]
    attnames = pycollect(o[:attrs])
    attvals  = [o[:attrs][:__getitem__](an) for an in attnames]
    ZarrArray{t,length(s)}(o,s,Dict(zip(attnames,attvals)))
  else
    error("Could not recognize zarr Object")
  end
end
jlind2pyind(::Colon)=pysli(nothing, nothing, nothing)
jlind2pyind(r::UnitRange)=pysli(first(r)-1,last(r),nothing)
jlind2pyind(i::Integer)=i-1

function finalizeResults(z::ZarrArray,ar::PyArray)
  ar2 = zeros(eltype(ar),reverse(size(ar)))
  permutedims!(ar2,ar,ntuple(i->ndims(ar)-i+1,ndims(ar)))
  ar2
end
finalizeResults(z::ZarrArray,ar::PyArray{<:Any,0})=ar[1]
finalizeResults(z::ZarrArray{<:Any,1},ar::PyVector)=ar
finalizeResults(z::ZarrArray{<:Any,1},ar::PyArray{<:Any,0})=ar[1]

function Base.getindex(z::ZarrArray,i...)
  pyinds=reverse(map(jlind2pyind,i))
  ar = pycall(z.v[:__getitem__],PyArray,pyinds)
  finalizeResults(z,ar)
end
Base.length(z::ZarrArray)=prod(z.s)
Base.size(z::ZarrArray)=z.s
Base.size(z::ZarrArray,i)=z.s[i]
Base.eltype{T}(z::ZarrArray{T})=T


function zopen(p, mode="r")
  zarrwrap(zarr.open(p))
end

end # module
