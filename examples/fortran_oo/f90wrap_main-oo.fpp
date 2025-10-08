# 1 "/home/ert/code/f90wrap/examples/fortran_oo/f90wrap_main-oo.f90"
# 1 "<built-in>"
# 1 "<command-line>"
# 1 "/home/ert/code/f90wrap/examples/fortran_oo/f90wrap_main-oo.f90"
! Module m_geometry defined in file main-oo.f90

subroutine f90wrap_rectangle__get__length(this, f90wrap_length)
    use m_geometry, only: rectangle
    implicit none
    type rectangle_wrapper_type
        class(rectangle), allocatable :: obj
    end type rectangle_wrapper_type
    type rectangle_ptr_type
        class(rectangle_wrapper_type), pointer :: p => NULL()
    end type rectangle_ptr_type
    integer, intent(in)   :: this(4)
    type(Rectangle_ptr_type) :: this_ptr
    real(4), intent(out) :: f90wrap_length
    
    this_ptr = transfer(this, this_ptr)
    f90wrap_length = this_ptr%p%obj%length
end subroutine f90wrap_rectangle__get__length

subroutine f90wrap_rectangle__set__length(this, f90wrap_length)
    use m_geometry, only: rectangle
    implicit none
    type rectangle_wrapper_type
        class(rectangle), allocatable :: obj
    end type rectangle_wrapper_type
    type rectangle_ptr_type
        class(rectangle_wrapper_type), pointer :: p => NULL()
    end type rectangle_ptr_type
    integer, intent(in)   :: this(4)
    type(Rectangle_ptr_type) :: this_ptr
    real(4), intent(in) :: f90wrap_length
    
    this_ptr = transfer(this, this_ptr)
    this_ptr%p%obj%length = f90wrap_length
end subroutine f90wrap_rectangle__set__length

subroutine f90wrap_rectangle__get__width(this, f90wrap_width)
    use m_geometry, only: rectangle
    implicit none
    type rectangle_wrapper_type
        class(rectangle), allocatable :: obj
    end type rectangle_wrapper_type
    type rectangle_ptr_type
        class(rectangle_wrapper_type), pointer :: p => NULL()
    end type rectangle_ptr_type
    integer, intent(in)   :: this(4)
    type(Rectangle_ptr_type) :: this_ptr
    real(4), intent(out) :: f90wrap_width
    
    this_ptr = transfer(this, this_ptr)
    f90wrap_width = this_ptr%p%obj%width
end subroutine f90wrap_rectangle__get__width

subroutine f90wrap_rectangle__set__width(this, f90wrap_width)
    use m_geometry, only: rectangle
    implicit none
    type rectangle_wrapper_type
        class(rectangle), allocatable :: obj
    end type rectangle_wrapper_type
    type rectangle_ptr_type
        class(rectangle_wrapper_type), pointer :: p => NULL()
    end type rectangle_ptr_type
    integer, intent(in)   :: this(4)
    type(Rectangle_ptr_type) :: this_ptr
    real(4), intent(in) :: f90wrap_width
    
    this_ptr = transfer(this, this_ptr)
    this_ptr%p%obj%width = f90wrap_width
end subroutine f90wrap_rectangle__set__width

subroutine f90wrap_m_geometry__perimeter__binding__rectangle(this, ret_perimeter)
    use m_geometry, only: rectangle
    implicit none
    
    type rectangle_wrapper_type
        class(rectangle), allocatable :: obj
    end type rectangle_wrapper_type
    type rectangle_ptr_type
        class(rectangle_wrapper_type), pointer :: p => NULL()
    end type rectangle_ptr_type
    type(rectangle_ptr_type) :: this_ptr
    integer, intent(in), dimension(4) :: this
    real(4), intent(out) :: ret_perimeter
    this_ptr = transfer(this, this_ptr)
    ret_perimeter = this_ptr%p%obj%perimeter()
end subroutine f90wrap_m_geometry__perimeter__binding__rectangle

subroutine f90wrap_m_geometry__is_square__binding__rectangle(this, ret_is_square)
    use m_geometry, only: rectangle
    implicit none
    
    type rectangle_wrapper_type
        class(rectangle), allocatable :: obj
    end type rectangle_wrapper_type
    type rectangle_ptr_type
        class(rectangle_wrapper_type), pointer :: p => NULL()
    end type rectangle_ptr_type
    type(rectangle_ptr_type) :: this_ptr
    integer, intent(in), dimension(4) :: this
    integer, intent(out) :: ret_is_square
    this_ptr = transfer(this, this_ptr)
    ret_is_square = this_ptr%p%obj%is_square()
end subroutine f90wrap_m_geometry__is_square__binding__rectangle

subroutine f90wrap_m_geometry__area__binding__rectangle(this, ret_area)
    use m_geometry, only: rectangle
    implicit none
    
    type rectangle_wrapper_type
        class(rectangle), allocatable :: obj
    end type rectangle_wrapper_type
    type rectangle_ptr_type
        class(rectangle_wrapper_type), pointer :: p => NULL()
    end type rectangle_ptr_type
    type(rectangle_ptr_type) :: this_ptr
    integer, intent(in), dimension(4) :: this
    real(4), intent(out) :: ret_area
    this_ptr = transfer(this, this_ptr)
    ret_area = this_ptr%p%obj%area()
end subroutine f90wrap_m_geometry__area__binding__rectangle

subroutine f90wrap_m_base_poly__is_polygone__binding__polygone_rectangle(this, ret_is_polygone)
    use m_geometry, only: rectangle
    implicit none
    
    type rectangle_wrapper_type
        class(rectangle), allocatable :: obj
    end type rectangle_wrapper_type
    type rectangle_ptr_type
        class(rectangle_wrapper_type), pointer :: p => NULL()
    end type rectangle_ptr_type
    type(rectangle_ptr_type) :: this_ptr
    integer, intent(in), dimension(4) :: this
    integer, intent(out) :: ret_is_polygone
    this_ptr = transfer(this, this_ptr)
    ret_is_polygone = this_ptr%p%obj%is_polygone()
end subroutine f90wrap_m_base_poly__is_polygone__binding__polygone_rectangle

subroutine f90wrap_m_geometry__construct_square(ret_construct_square, length)
    use m_geometry, only: square
    implicit none
    
    type square_wrapper_type
        class(square), allocatable :: obj
    end type square_wrapper_type
    type square_ptr_type
        type(square_wrapper_type), pointer :: p => NULL()
    end type square_ptr_type
    type(square_ptr_type) :: ret_construct_square_ptr
    integer, intent(out), dimension(4) :: ret_construct_square
    real, intent(in) :: length
    allocate(ret_construct_square_ptr%p)
    ret_construct_square_ptr%p%obj = square(length=length)
    ret_construct_square = transfer(ret_construct_square_ptr, ret_construct_square)
end subroutine f90wrap_m_geometry__construct_square

subroutine f90wrap_m_geometry__square_finalise(this)
    use m_geometry, only: square
    implicit none
    
    type square_wrapper_type
        class(square), allocatable :: obj
    end type square_wrapper_type
    type square_ptr_type
        type(square_wrapper_type), pointer :: p => NULL()
    end type square_ptr_type
    type(square_ptr_type) :: this_ptr
    integer, intent(in), dimension(4) :: this
    this_ptr = transfer(this, this_ptr)
    deallocate(this_ptr%p)
end subroutine f90wrap_m_geometry__square_finalise

subroutine f90wrap_m_geometry__is_square__binding__square(this, ret_is_square)
    use m_geometry, only: square
    implicit none
    
    type square_wrapper_type
        class(square), allocatable :: obj
    end type square_wrapper_type
    type square_ptr_type
        type(square_wrapper_type), pointer :: p => NULL()
    end type square_ptr_type
    type(square_ptr_type) :: this_ptr
    integer, intent(in), dimension(4) :: this
    integer, intent(out) :: ret_is_square
    this_ptr = transfer(this, this_ptr)
    ret_is_square = this_ptr%p%obj%is_square()
end subroutine f90wrap_m_geometry__is_square__binding__square

subroutine f90wrap_m_geometry__area__binding__square(this, ret_area)
    use m_geometry, only: square
    implicit none
    
    type square_wrapper_type
        class(square), allocatable :: obj
    end type square_wrapper_type
    type square_ptr_type
        type(square_wrapper_type), pointer :: p => NULL()
    end type square_ptr_type
    type(square_ptr_type) :: this_ptr
    integer, intent(in), dimension(4) :: this
    real(4), intent(out) :: ret_area
    this_ptr = transfer(this, this_ptr)
    ret_area = this_ptr%p%obj%area()
end subroutine f90wrap_m_geometry__area__binding__square

subroutine f90wrap_m_geometry__perimeter__binding__rectangle_square(this, ret_perimeter)
    use m_geometry, only: square
    implicit none
    
    type square_wrapper_type
        class(square), allocatable :: obj
    end type square_wrapper_type
    type square_ptr_type
        type(square_wrapper_type), pointer :: p => NULL()
    end type square_ptr_type
    type(square_ptr_type) :: this_ptr
    integer, intent(in), dimension(4) :: this
    real(4), intent(out) :: ret_perimeter
    this_ptr = transfer(this, this_ptr)
    ret_perimeter = this_ptr%p%obj%perimeter()
end subroutine f90wrap_m_geometry__perimeter__binding__rectangle_square

subroutine f90wrap_m_base_poly__is_polygone__binding__polygone_rectang5400(this, ret_is_polygone)
    use m_geometry, only: square
    implicit none
    
    type square_wrapper_type
        class(square), allocatable :: obj
    end type square_wrapper_type
    type square_ptr_type
        type(square_wrapper_type), pointer :: p => NULL()
    end type square_ptr_type
    type(square_ptr_type) :: this_ptr
    integer, intent(in), dimension(4) :: this
    integer, intent(out) :: ret_is_polygone
    this_ptr = transfer(this, this_ptr)
    ret_is_polygone = this_ptr%p%obj%is_polygone()
end subroutine f90wrap_m_base_poly__is_polygone__binding__polygone_rectang5400

subroutine f90wrap_circle__get__radius(this, f90wrap_radius)
    use m_geometry, only: circle
    implicit none
    type circle_wrapper_type
        class(circle), allocatable :: obj
    end type circle_wrapper_type
    type circle_ptr_type
        type(circle_wrapper_type), pointer :: p => NULL()
    end type circle_ptr_type
    integer, intent(in)   :: this(4)
    type(Circle_ptr_type) :: this_ptr
    real(4), intent(out) :: f90wrap_radius
    
    this_ptr = transfer(this, this_ptr)
    f90wrap_radius = this_ptr%p%obj%radius
end subroutine f90wrap_circle__get__radius

subroutine f90wrap_circle__set__radius(this, f90wrap_radius)
    use m_geometry, only: circle
    implicit none
    type circle_wrapper_type
        class(circle), allocatable :: obj
    end type circle_wrapper_type
    type circle_ptr_type
        type(circle_wrapper_type), pointer :: p => NULL()
    end type circle_ptr_type
    integer, intent(in)   :: this(4)
    type(Circle_ptr_type) :: this_ptr
    real(4), intent(in) :: f90wrap_radius
    
    this_ptr = transfer(this, this_ptr)
    this_ptr%p%obj%radius = f90wrap_radius
end subroutine f90wrap_circle__set__radius

subroutine f90wrap_m_geometry__construct_circle(ret_construct_circle, rc, rb)
    use m_geometry, only: circle
    implicit none
    
    type circle_wrapper_type
        class(circle), allocatable :: obj
    end type circle_wrapper_type
    type circle_ptr_type
        type(circle_wrapper_type), pointer :: p => NULL()
    end type circle_ptr_type
    type(circle_ptr_type) :: ret_construct_circle_ptr
    integer, intent(out), dimension(4) :: ret_construct_circle
    real, intent(in) :: rc
    real, intent(in) :: rb
    allocate(ret_construct_circle_ptr%p)
    ret_construct_circle_ptr%p%obj = circle(rc=rc, rb=rb)
    ret_construct_circle = transfer(ret_construct_circle_ptr, ret_construct_circle)
end subroutine f90wrap_m_geometry__construct_circle

subroutine f90wrap_m_geometry__area__binding__circle(this, ret_area)
    use m_geometry, only: circle
    implicit none
    
    type circle_wrapper_type
        class(circle), allocatable :: obj
    end type circle_wrapper_type
    type circle_ptr_type
        type(circle_wrapper_type), pointer :: p => NULL()
    end type circle_ptr_type
    type(circle_ptr_type) :: this_ptr
    integer, intent(in), dimension(4) :: this
    real(4), intent(out) :: ret_area
    this_ptr = transfer(this, this_ptr)
    ret_area = this_ptr%p%obj%area()
end subroutine f90wrap_m_geometry__area__binding__circle

subroutine f90wrap_m_geometry__print__binding__circle(this)
    use m_geometry, only: circle
    implicit none
    
    type circle_wrapper_type
        class(circle), allocatable :: obj
    end type circle_wrapper_type
    type circle_ptr_type
        type(circle_wrapper_type), pointer :: p => NULL()
    end type circle_ptr_type
    type(circle_ptr_type) :: this_ptr
    integer, intent(in), dimension(4) :: this
    this_ptr = transfer(this, this_ptr)
    call this_ptr%p%obj%print()
end subroutine f90wrap_m_geometry__print__binding__circle

subroutine f90wrap_m_geometry__obj_name__binding__circle(obj)
    use m_geometry, only: circle
    implicit none
    
    type circle_wrapper_type
        class(circle), allocatable :: obj
    end type circle_wrapper_type
    type circle_ptr_type
        type(circle_wrapper_type), pointer :: p => NULL()
    end type circle_ptr_type
    type(circle_ptr_type) :: obj_ptr
    integer, intent(in), dimension(4) :: obj
    obj_ptr = transfer(obj, obj_ptr)
    call obj_ptr%p%obj%obj_name()
end subroutine f90wrap_m_geometry__obj_name__binding__circle

subroutine f90wrap_m_geometry__copy__binding__circle(this, from_)
    use m_geometry, only: circle
    implicit none
    
    type circle_wrapper_type
        class(circle), allocatable :: obj
    end type circle_wrapper_type
    type circle_ptr_type
        type(circle_wrapper_type), pointer :: p => NULL()
    end type circle_ptr_type
    type(circle_ptr_type) :: this_ptr
    integer, intent(in), dimension(4) :: this
    type(circle_ptr_type) :: from__ptr
    integer, intent(in), dimension(4) :: from_
    this_ptr = transfer(this, this_ptr)
    from__ptr = transfer(from_, from__ptr)
    call this_ptr%p%obj%copy(from=from__ptr%p%obj)
end subroutine f90wrap_m_geometry__copy__binding__circle

subroutine f90wrap_m_geometry__private_method__binding__circle(this)
    use m_geometry, only: circle
    implicit none
    
    type circle_wrapper_type
        class(circle), allocatable :: obj
    end type circle_wrapper_type
    type circle_ptr_type
        type(circle_wrapper_type), pointer :: p => NULL()
    end type circle_ptr_type
    type(circle_ptr_type) :: this_ptr
    integer, intent(in), dimension(4) :: this
    this_ptr = transfer(this, this_ptr)
    call this_ptr%p%obj%private_method()
end subroutine f90wrap_m_geometry__private_method__binding__circle

subroutine f90wrap_m_geometry__perimeter_4__binding__circle(this, radius, ret_perimeter)
    use m_geometry, only: circle
    implicit none
    
    type circle_wrapper_type
        class(circle), allocatable :: obj
    end type circle_wrapper_type
    type circle_ptr_type
        type(circle_wrapper_type), pointer :: p => NULL()
    end type circle_ptr_type
    type(circle_ptr_type) :: this_ptr
    integer, intent(in), dimension(4) :: this
    real(4), intent(in) :: radius
    real(4), intent(out) :: ret_perimeter
    this_ptr = transfer(this, this_ptr)
    ret_perimeter = this_ptr%p%obj%perimeter_4(radius=radius)
end subroutine f90wrap_m_geometry__perimeter_4__binding__circle

subroutine f90wrap_m_geometry__perimeter_8__binding__circle(this, radius, ret_perimeter)
    use m_geometry, only: circle
    implicit none
    
    type circle_wrapper_type
        class(circle), allocatable :: obj
    end type circle_wrapper_type
    type circle_ptr_type
        type(circle_wrapper_type), pointer :: p => NULL()
    end type circle_ptr_type
    type(circle_ptr_type) :: this_ptr
    integer, intent(in), dimension(4) :: this
    real(8), intent(in) :: radius
    real(8), intent(out) :: ret_perimeter
    this_ptr = transfer(this, this_ptr)
    ret_perimeter = this_ptr%p%obj%perimeter_8(radius=radius)
end subroutine f90wrap_m_geometry__perimeter_8__binding__circle

subroutine f90wrap_m_geometry__circle_free__binding__circle(this)
    use m_geometry, only: circle, circle_free
    implicit none
    
    type circle_wrapper_type
        class(circle), allocatable :: obj
    end type circle_wrapper_type
    type circle_ptr_type
        type(circle_wrapper_type), pointer :: p => NULL()
    end type circle_ptr_type
    type(circle_ptr_type) :: this_ptr
    integer, intent(in), dimension(4) :: this
    this_ptr = transfer(this, this_ptr)
    call circle_free(this=this_ptr%p%obj)
    deallocate(this_ptr%p)
end subroutine f90wrap_m_geometry__circle_free__binding__circle

subroutine f90wrap_m_geometry__construct_ball(ret_construct_ball, rc, rb)
    use m_geometry, only: ball
    implicit none
    
    type ball_wrapper_type
        class(ball), allocatable :: obj
    end type ball_wrapper_type
    type ball_ptr_type
        type(ball_wrapper_type), pointer :: p => NULL()
    end type ball_ptr_type
    type(ball_ptr_type) :: ret_construct_ball_ptr
    integer, intent(out), dimension(4) :: ret_construct_ball
    real, intent(in) :: rc
    real, intent(in) :: rb
    allocate(ret_construct_ball_ptr%p)
    ret_construct_ball_ptr%p%obj = ball(rc=rc, rb=rb)
    ret_construct_ball = transfer(ret_construct_ball_ptr, ret_construct_ball)
end subroutine f90wrap_m_geometry__construct_ball

subroutine f90wrap_m_geometry__ball_finalise(this)
    use m_geometry, only: ball
    implicit none
    
    type ball_wrapper_type
        class(ball), allocatable :: obj
    end type ball_wrapper_type
    type ball_ptr_type
        type(ball_wrapper_type), pointer :: p => NULL()
    end type ball_ptr_type
    type(ball_ptr_type) :: this_ptr
    integer, intent(in), dimension(4) :: this
    this_ptr = transfer(this, this_ptr)
    deallocate(this_ptr%p)
end subroutine f90wrap_m_geometry__ball_finalise

subroutine f90wrap_m_geometry__volume__binding__ball(this, ret_volume)
    use m_geometry, only: ball
    implicit none
    
    type ball_wrapper_type
        class(ball), allocatable :: obj
    end type ball_wrapper_type
    type ball_ptr_type
        type(ball_wrapper_type), pointer :: p => NULL()
    end type ball_ptr_type
    type(ball_ptr_type) :: this_ptr
    integer, intent(in), dimension(4) :: this
    real(4), intent(out) :: ret_volume
    this_ptr = transfer(this, this_ptr)
    ret_volume = this_ptr%p%obj%volume()
end subroutine f90wrap_m_geometry__volume__binding__ball

subroutine f90wrap_m_geometry__area__binding__ball(this, ret_area)
    use m_geometry, only: ball
    implicit none
    
    type ball_wrapper_type
        class(ball), allocatable :: obj
    end type ball_wrapper_type
    type ball_ptr_type
        type(ball_wrapper_type), pointer :: p => NULL()
    end type ball_ptr_type
    type(ball_ptr_type) :: this_ptr
    integer, intent(in), dimension(4) :: this
    real(4), intent(out) :: ret_area
    this_ptr = transfer(this, this_ptr)
    ret_area = this_ptr%p%obj%area()
end subroutine f90wrap_m_geometry__area__binding__ball

subroutine f90wrap_m_geometry__private_method__binding__ball(this)
    use m_geometry, only: ball
    implicit none
    
    type ball_wrapper_type
        class(ball), allocatable :: obj
    end type ball_wrapper_type
    type ball_ptr_type
        type(ball_wrapper_type), pointer :: p => NULL()
    end type ball_ptr_type
    type(ball_ptr_type) :: this_ptr
    integer, intent(in), dimension(4) :: this
    this_ptr = transfer(this, this_ptr)
    call this_ptr%p%obj%private_method()
end subroutine f90wrap_m_geometry__private_method__binding__ball

subroutine f90wrap_m_geometry__circle_print(this)
    use m_geometry, only: circle_print, circle
    implicit none
    
    type circle_wrapper_type
        class(circle), allocatable :: obj
    end type circle_wrapper_type
    type circle_ptr_type
        type(circle_wrapper_type), pointer :: p => NULL()
    end type circle_ptr_type
    type(circle_ptr_type) :: this_ptr
    integer, intent(in), dimension(4) :: this
    this_ptr = transfer(this, this_ptr)
    call circle_print(this=this_ptr%p%obj)
end subroutine f90wrap_m_geometry__circle_print

subroutine f90wrap_m_geometry__circle_free(this)
    use m_geometry, only: circle, circle_free
    implicit none
    
    type circle_wrapper_type
        class(circle), allocatable :: obj
    end type circle_wrapper_type
    type circle_ptr_type
        type(circle_wrapper_type), pointer :: p => NULL()
    end type circle_ptr_type
    type(circle_ptr_type) :: this_ptr
    integer, intent(in), dimension(4) :: this
    this_ptr = transfer(this, this_ptr)
    call circle_free(this=this_ptr%p%obj)
end subroutine f90wrap_m_geometry__circle_free

subroutine f90wrap_m_geometry__ball_area(this, ret_area)
    use m_geometry, only: ball, ball_area
    implicit none
    
    type ball_wrapper_type
        class(ball), allocatable :: obj
    end type ball_wrapper_type
    type ball_ptr_type
        type(ball_wrapper_type), pointer :: p => NULL()
    end type ball_ptr_type
    type(ball_ptr_type) :: this_ptr
    integer, intent(in), dimension(4) :: this
    real(4), intent(out) :: ret_area
    this_ptr = transfer(this, this_ptr)
    ret_area = ball_area(this=this_ptr%p%obj)
end subroutine f90wrap_m_geometry__ball_area

subroutine f90wrap_m_geometry__get_ball_radius(my_ball, ret_radius)
    use m_geometry, only: get_ball_radius, ball
    implicit none
    
    type ball_wrapper_type
        class(ball), allocatable :: obj
    end type ball_wrapper_type
    type ball_ptr_type
        type(ball_wrapper_type), pointer :: p => NULL()
    end type ball_ptr_type
    type(ball_ptr_type) :: my_ball_ptr
    integer, intent(in), dimension(4) :: my_ball
    real(4), intent(out) :: ret_radius
    my_ball_ptr = transfer(my_ball, my_ball_ptr)
    ret_radius = get_ball_radius(my_ball=my_ball_ptr%p%obj)
end subroutine f90wrap_m_geometry__get_ball_radius

subroutine f90wrap_m_geometry__circle_copy(this, from_)
    use m_geometry, only: circle, circle_copy
    implicit none
    
    type circle_wrapper_type
        class(circle), allocatable :: obj
    end type circle_wrapper_type
    type circle_ptr_type
        type(circle_wrapper_type), pointer :: p => NULL()
    end type circle_ptr_type
    type(circle_ptr_type) :: this_ptr
    integer, intent(in), dimension(4) :: this
    type(circle_ptr_type) :: from__ptr
    integer, intent(in), dimension(4) :: from_
    this_ptr = transfer(this, this_ptr)
    from__ptr = transfer(from_, from__ptr)
    call circle_copy(this=this_ptr%p%obj, from=from__ptr%p%obj)
end subroutine f90wrap_m_geometry__circle_copy

subroutine f90wrap_m_geometry__ball_volume(this, ret_volume)
    use m_geometry, only: ball_volume, ball
    implicit none
    
    type ball_wrapper_type
        class(ball), allocatable :: obj
    end type ball_wrapper_type
    type ball_ptr_type
        type(ball_wrapper_type), pointer :: p => NULL()
    end type ball_ptr_type
    type(ball_ptr_type) :: this_ptr
    integer, intent(in), dimension(4) :: this
    real(4), intent(out) :: ret_volume
    this_ptr = transfer(this, this_ptr)
    ret_volume = ball_volume(this=this_ptr%p%obj)
end subroutine f90wrap_m_geometry__ball_volume

subroutine f90wrap_m_geometry__get_circle_radius(my_circle, ret_radius)
    use m_geometry, only: circle, get_circle_radius
    implicit none
    
    type circle_wrapper_type
        class(circle), allocatable :: obj
    end type circle_wrapper_type
    type circle_ptr_type
        type(circle_wrapper_type), pointer :: p => NULL()
    end type circle_ptr_type
    type(circle_ptr_type) :: my_circle_ptr
    integer, intent(in), dimension(4) :: my_circle
    real(4), intent(out) :: ret_radius
    my_circle_ptr = transfer(my_circle, my_circle_ptr)
    ret_radius = get_circle_radius(my_circle=my_circle_ptr%p%obj)
end subroutine f90wrap_m_geometry__get_circle_radius

subroutine f90wrap_m_geometry__circle_obj_name(obj)
    use m_geometry, only: circle, circle_obj_name
    implicit none
    
    type circle_wrapper_type
        class(circle), allocatable :: obj
    end type circle_wrapper_type
    type circle_ptr_type
        type(circle_wrapper_type), pointer :: p => NULL()
    end type circle_ptr_type
    type(circle_ptr_type) :: obj_ptr
    integer, intent(in), dimension(4) :: obj
    obj_ptr = transfer(obj, obj_ptr)
    call circle_obj_name(obj=obj_ptr%p%obj)
end subroutine f90wrap_m_geometry__circle_obj_name

subroutine f90wrap_m_geometry__circle_area(this, ret_area)
    use m_geometry, only: circle, circle_area
    implicit none
    
    type circle_wrapper_type
        class(circle), allocatable :: obj
    end type circle_wrapper_type
    type circle_ptr_type
        type(circle_wrapper_type), pointer :: p => NULL()
    end type circle_ptr_type
    type(circle_ptr_type) :: this_ptr
    integer, intent(in), dimension(4) :: this
    real(4), intent(out) :: ret_area
    this_ptr = transfer(this, this_ptr)
    ret_area = circle_area(this=this_ptr%p%obj)
end subroutine f90wrap_m_geometry__circle_area

subroutine f90wrap_m_geometry__get__pi(f90wrap_pi)
    use m_geometry, only: m_geometry_pi => pi
    implicit none
    real(4), intent(out) :: f90wrap_pi
    
    f90wrap_pi = m_geometry_pi
end subroutine f90wrap_m_geometry__get__pi

! End of module m_geometry defined in file main-oo.f90
