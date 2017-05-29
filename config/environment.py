# PLASMA is a software package provided by:
# University of Tennessee, US,
# University of Manchester, UK.

import os

# ------------------------------------------------------------------------------
class Environment:
	def __init__( self ):
		self.stack = [ os.environ, {} ]
	# end
	
	# push( self, env={} ) would use the same hash each time;
	# this pushes a new hash each time
	def push( self, env=None ):
		if (not env):
			env = {}
		self.stack.append( env )
	
	def top( self ):
		return self.stack[-1]
	
	def pop( self ):
		if (len(self.stack) == 2):
			raise Exception( "attempting to pop last user environment" )
		return self.stack.pop()
	
	# compared to __getitem__, returns None if key doesn't exist
	def get( self, key ):
		for env in self.stack[::-1]:
			if (env.has_key( key )):
				return env[key]
		return None
	
	def __getitem__( self, key ):
		for env in self.stack[::-1]:
			if (env.has_key( key )):
				return env[key]
		return '' # or None?
	
	# todo: should val = None delete the key?
	def __setitem__( self, key, val ):
		self.stack[-1][ key ] = val
	
	def append( self, key, val ):
		orig = self[ key ]  #self.get( key )
		if (val):
			if (orig):
				val = orig + ' ' + val
			self[key] = val
		return orig
	
	def prepend( self, key, val ):
		orig = self[ key ]  #self.get( key )
		if (val):
			if (orig):
				val = val + ' ' + orig
			self[key] = val
		return orig
# end

# ------------------------------------------------------------------------------
def test():
	env = Environment()
	print env.stack
	print
	
	CC  = env['CC']
	CXX = env['CXX']
	print 'CC  <' + CC  + '>'
	print 'CXX <' + CXX + '>'
	print
	
	env.push()
	print env.stack
	print
	
	env['CC'] = 'icc'
	print env.stack
	print
	
	CC = env['CC']
	env['CC'] += 'foo'
	print env.stack
	print
	
	CXX = env['CXX']
	env['CXX'] += 'foo'
	print env.stack
	print
	
	save_CFLAGS = env.prepend( 'CFLAGS', '-O2' )
	print env.stack
	env['CFLAGS'] = save_CFLAGS
	print env.stack
	print
	
	save_CFLAGS = env.append( 'CFLAGS', '-O2' )
	print env.stack
	env['CFLAGS'] = save_CFLAGS
	print env.stack
	print
	
	save_CXXFLAGS = env.prepend( 'CXXFLAGS', '-O2' )
	print env.stack
	env['CXXFLAGS'] = save_CXXFLAGS
	print env.stack
	print
	
	save_CXXFLAGS = env.prepend( 'CXXFLAGS', '-g' )
	print env.stack
	print
	
	save_CXXFLAGS = env.append( 'CXXFLAGS', '-Wshadow' )
	print env.stack
	env['CXXFLAGS'] = save_CXXFLAGS
	print env.stack
	print
# end

# ------------------------------------------------------------------------------
if (__name__ == '__main__'):
	test()
