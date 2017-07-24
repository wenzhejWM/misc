%module(directors="1") testSwig    // "testSwig" is the name of the module you will load in Java

//this will be included in generated java code
%pragma(java) jniclasscode=%{
	static {
		try {
			System.loadLibrary("testSwig");
		} catch (UnsatisfiedLinkError e) {
			throw new RuntimeException(e);
		}
	}
%}

// ************************************* Configuration for exception handling *********************************
%include "std_string.i"
%include "exception.i"

// myException derives from std::runtime_error but SWIG does not recognize std::runtime_error
// A hack to resolve this is to "redefine" runtime_error within std namespace
// the effect is that SWIG will write std::runtime_error in the wrapper.cxx file and the real implementation of runtime_error
// provided by <stdexcept> will be used.

namespace std {
  struct exception 
  {
    virtual ~exception() throw();
    virtual const char* what() const throw();
  };
  struct runtime_error : exception 
  {
    runtime_error(const string& msg);
  };
}

// When a new type of exception is added, the error handler in the macro below has to be extended 
// to catch that type of exception

%define WRAP_THROWN_EXCEPTION(FUNC)
	%exception func {
		try{
			$action
		}catch(myException& e){
			jclass eclass = jenv->FindClass("java/lang/RuntimeException");  
			if ( eclass ) {
				jenv->ThrowNew( eclass, e.what()); 
			}
		}
	}
%enddef



// If a C++ function is expected to throw exceptions, a line like the one below has to be given 
//in this configuration file to let SWIG catch the exception and rethrow its Java counterpart. 
// This is to avoid the try/catch part defined above being put in every function in wrapper.cxx file.

WRAP_THROWN_EXCEPTION(func);

// ************************************* End of configuration for exception handling **************************

//this will be included in generated c++ code
%inline %{
#include "func.h"
#include "myException.h"
%}


// include the header files again for SWIG (with %)
%include "func.h"
%include "myException.h"


