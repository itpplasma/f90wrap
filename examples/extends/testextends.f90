
module testextends_mod

implicit none
public

	! -----------------------------------------------
	type, abstract :: Superclass
		! IN: Ask subroutine to stop in the middle.
		integer :: stop_at = -1		! -1 --> don't stop
		! contains
		! 	procedure :: get_value
	end type

	type, extends(Superclass) :: Subclass1
		integer :: nl
	end type

	type, extends(Superclass) :: Subclass2
		integer :: nl
		! contains
		! 	procedure :: get_value => get_nl
	end type

contains

	! function get_value(self) result(res)
	! 	implicit none
	! 	class(Superclass), intent(in) :: self
	! 	integer :: res

	! 	res = self%stop_at
	! end function

	! function get_nl(self) result(res)
	! 	implicit none
	! 	class(Subclass2), intent(in) :: self
	! 	integer :: res

	! 	res = self%nl
	! end function
end module
