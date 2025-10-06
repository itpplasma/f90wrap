module dnad
  implicit none
contains
  function abs_d(x) result(y)
    real(8), intent(in) :: x
    real(8) :: y
    y = abs(x)
  end function abs_d

  subroutine add_di(x, i)
    real(8), intent(inout) :: x
    integer, intent(in) :: i
    x = x + real(i, kind=8)
  end subroutine add_di

  subroutine assign_di(x, i)
    real(8), intent(out) :: x
    integer, intent(in) :: i
    x = real(i, kind=8)
  end subroutine assign_di
end module dnad
